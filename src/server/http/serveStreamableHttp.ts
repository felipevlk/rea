import {
  createMcpHandler,
  hostHeaderValidationResponse,
  localhostAllowedHostnames,
  originValidationResponse,
  type McpHttpHandler,
  type McpRequestContext,
} from "@modelcontextprotocol/server";
import {
  createServer as createHttpServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { isIPv4, type AddressInfo } from "node:net";
import type { Logger } from "pino";
import {
  normalizeHttpHostname,
  type HttpHostname,
} from "../../domain/httpHostname.js";

import {
  toWebRequest,
  UnsupportedRequestTargetError,
  writeWebResponse,
} from "./nodeHttpBridge.js";
import { listenHttpServer } from "./listenHttpServer.js";
import {
  StreamableHttpHostAllowlistError,
  StreamableHttpListenError,
} from "./streamableHttpListenError.js";
import {
  NodeHttpRequestOwner,
  type HttpAdmissionLifetime,
  type NodeHttpExchangeContext,
} from "./NodeHttpRequestOwner.js";
import type { RequestTrackingMcpServer } from "../RequestTrackingMcpServer.js";
import { errorMessage } from "../../kernel/errorMessage.js";

/** MCP endpoint path shared by `rea mcp --http` and agent environment servers. */
export const MCP_HTTP_PATH = "/mcp";

const DEFAULT_REQUEST_DRAIN_TIMEOUT_MS = 10_000;

/** One plain HTTP route served beside the MCP endpoint as web Request to Response. */
export interface StreamableHttpRoute {
  readonly method: string;
  readonly path: string;
  readonly handle: (request: Request) => Response | Promise<Response>;
}

export interface ServeStreamableHttpOptions {
  readonly host: string;
  readonly port: number;
  /** Operator-declared Host/Origin names; a non-loopback bind refuses to serve without them. */
  readonly allowedHosts?: readonly HttpHostname[];
  /** Cancellation during listener startup; later shutdown remains owned by close(). */
  readonly signal?: AbortSignal;
  /** Called once per modern request; the handler drains and closes each instance. */
  readonly createMcpServer: (
    context: McpRequestContext,
  ) => RequestTrackingMcpServer | Promise<RequestTrackingMcpServer>;
  /** Maximum time a closing request instance waits for its tool handlers (default 10s). */
  readonly requestDrainTimeoutMs?: number;
  /** Concurrent HTTP input/route work (default 8); unread uploads retain their permit. */
  readonly maxConcurrentDispatches?: number;
  readonly routes?: readonly StreamableHttpRoute[];
  readonly logger?: Logger;
  /** Reports listener failures after binding; request failures are logged instead. */
  readonly onerror?: (error: StreamableHttpListenError) => void;
}

export interface StreamableHttpServer {
  /** The bound MCP endpoint URL. */
  readonly url: string;
  /** Stop incoming dispatch and drain handlers before shared provider owners close. */
  readonly prepareShutdown: () => Promise<void>;
  readonly close: () => Promise<void>;
}

const isLoopbackIpv4 = (address: string): boolean =>
  isIPv4(address) && address.startsWith("127.");
/** Classify the native bound IP; configured DNS names are not binding evidence. */
const isLoopbackAddress = (hostname: string): boolean => {
  if (isLoopbackIpv4(hostname)) return true;
  if (!hostname.startsWith("[") || !hostname.endsWith("]")) return false;
  const inner = hostname.slice(1, -1);
  if (inner === "::1") return true;
  if (!inner.startsWith("::ffff:")) return false;
  const mapped = inner.slice("::ffff:".length);
  if (isLoopbackIpv4(mapped)) return true;
  const hex = /^([0-9a-f]{1,4}):[0-9a-f]{1,4}$/u.exec(mapped);
  return hex !== null && Number.parseInt(hex[1] ?? "0", 16) >> 8 === 0x7f;
};
/** Classify the bound address first, then allow its address and configured name. */
export const hostHeaderAllowlist = (
  host: string,
  boundAddress: string,
): string[] | undefined => {
  const bound = normalizeHttpHostname(boundAddress);
  if (bound === undefined || !isLoopbackAddress(bound)) return undefined;
  const names = [normalizeHttpHostname(host), bound];
  const known = names.filter((name): name is string => name !== undefined);
  return [...new Set([...localhostAllowedHostnames(), ...known])];
};

/** A JSON-RPC error answered outside any request, so its id is null. */
export const jsonRpcErrorResponse = (
  status: number,
  code: number,
  message: string,
): Response =>
  Response.json(
    { jsonrpc: "2.0", error: { code, message }, id: null },
    { status },
  );
const jsonError = jsonRpcErrorResponse;

const endpointUrl = (address: AddressInfo): string => {
  const host =
    address.family === "IPv6" ? `[${address.address}]` : address.address;
  return `http://${host}:${String(address.port)}${MCP_HTTP_PATH}`;
};

interface NodeRequestTarget {
  readonly route: (request: Request) => Promise<{
    readonly response: Response;
    readonly admissionLifetime: HttpAdmissionLifetime;
  }>;
  readonly origin: () => string;
  readonly logger: Logger | undefined;
}

/** Serve one MCP 2026-07-28 server, plus plain routes, over Streamable HTTP until closed. */
export const serveStreamableHttp = async (
  options: ServeStreamableHttpOptions,
): Promise<StreamableHttpServer> => {
  const { host, port, logger, routes = [] } = options;
  // Scoped IPv6 literals bind but cannot be named in a URL; refuse them before anything exists.
  if (host.includes("%"))
    throw new StreamableHttpListenError(
      new Error(`scoped IPv6 bind address ${host} is not supported`),
    );

  // Modern-only: createMcpHandler rejects 2025-era initialize/session traffic with
  // UnsupportedProtocolVersion naming the server's supported revisions. Each
  // request owns one instance, so cancellation waits for its tool handlers
  // before the instance's provider hooks run.
  const requestDrainTimeoutMs =
    options.requestDrainTimeoutMs ?? DEFAULT_REQUEST_DRAIN_TIMEOUT_MS;
  /** Instances whose exchange ended without closing them, drained at shutdown. */
  const products = new Set<RequestTrackingMcpServer>();
  const modern = createMcpHandler(
    async (context) => {
      const product = await options.createMcpServer(context);
      products.add(product);
      product.enableRequestTracking();
      const drain = async (): Promise<void> => {
        if (await product.drainRequests(requestDrainTimeoutMs)) return;
        logger?.warn(
          { timeout_ms: requestDrainTimeoutMs },
          "MCP request handlers still running when their instance closed",
        );
      };
      // An aborted exchange closes the protocol server directly, without
      // `product.close()`. Wrap that path too, so a cancelled tool or prompt
      // completion reaches its finally block before REA's provider hooks run.
      const protocol = product.server;
      const closeProtocol = protocol.close.bind(protocol);
      protocol.close = async () => {
        await drain();
        await closeProtocol();
      };
      let closing = false;
      const close = product.close.bind(product);
      product.close = async () => {
        closing = true;
        await drain();
        await close();
        products.delete(product);
      };
      // A disconnected exchange closes the transport without closing its
      // instance. Retire that instance now instead of retaining it until
      // process shutdown.
      const previousOnclose = protocol.onclose;
      protocol.onclose = () => {
        previousOnclose?.();
        if (closing) return;
        void product.close().catch((cause: unknown) => {
          logger?.warn(
            { error: errorMessage(cause) },
            "MCP request instance cleanup failed",
          );
        });
      };
      return product;
    },
    { legacy: "reject" },
  );

  let allowedHosts: string[] | undefined;
  let origin = "";
  let stopped = false;
  const route: NodeRequestTarget["route"] = async (request) => {
    const local = (response: Response) => ({
      response,
      admissionLifetime: "response" as const,
    });
    if (stopped) return local(jsonError(503, -32000, "Server shutting down"));
    // One resolved allowlist gates Host (DNS rebinding) and Origin (browser pages); a missing Origin passes.
    const refused =
      allowedHosts === undefined
        ? undefined
        : (hostHeaderValidationResponse(request, allowedHosts) ??
          originValidationResponse(request, allowedHosts));
    if (refused !== undefined) return local(refused);

    const { pathname } = new URL(request.url);
    if (pathname === MCP_HTTP_PATH)
      return {
        response: await modern.fetch(request),
        admissionLifetime: "dispatch",
      };
    const match = routes.find(
      (candidate) =>
        candidate.path === pathname && candidate.method === request.method,
    );
    return local(
      match === undefined
        ? jsonError(404, -32601, `No route for ${request.method} ${pathname}`)
        : await match.handle(request),
    );
  };

  const httpServer = createHttpServer();
  const target: NodeRequestTarget = { route, origin: () => origin, logger };
  const requests = new NodeHttpRequestOwner(
    options.maxConcurrentDispatches ?? 8,
    {
      serve: (request, response, context) =>
        serveNodeRequest(target, request, response, context),
      refuse: (reason) =>
        jsonError(
          503,
          -32000,
          reason === "capacity"
            ? "HTTP request capacity is full; retry after pending dispatches finish"
            : "HTTP response cleanup failed; this server is not accepting new requests",
        ),
    },
  );

  const prepareShutdown = stagedClose([
    async () => {
      stopped = true;
      await closeHttpServer(httpServer);
    },
    () => requests.drain(),
    () => modern.close(),
    async () => {
      // A client that disconnects removes its instance from the handler's
      // in-flight set without closing it. Drain and close those leftovers so
      // their cancelled handlers finish before their provider hooks run.
      const results = await Promise.allSettled(
        [...products].map((product) => product.close()),
      );
      const failures = results.flatMap((result) =>
        result.status === "rejected" ? [result.reason] : [],
      );
      if (failures.length === 1) throw failures[0];
      if (failures.length > 1)
        throw new AggregateError(
          failures,
          "MCP request instance cleanup failed",
        );
    },
  ]);

  try {
    await listenHttpServer(httpServer, host, port, options.signal);
    options.signal?.throwIfAborted();
  } catch (cause: unknown) {
    if (options.signal?.aborted && cause === options.signal.reason) {
      if (httpServer.listening) await prepareShutdown();
      throw cause;
    }
    throw new StreamableHttpListenError(cause);
  }

  const bound = httpServer.address();
  let url: string;
  try {
    if (bound === null || typeof bound === "string")
      throw new TypeError("MCP HTTP listener has no TCP address");
    const endpoint = new URL(endpointUrl(bound));
    const loopbackHosts = hostHeaderAllowlist(host, bound.address);
    if (loopbackHosts !== undefined) {
      allowedHosts = loopbackHosts;
    } else {
      // Configuration supplies canonical names; the listener does not reinterpret them.
      if (
        options.allowedHosts === undefined ||
        options.allowedHosts.length === 0
      )
        throw new StreamableHttpHostAllowlistError(bound.address);
      allowedHosts = [...options.allowedHosts];
    }
    const announcedHost = allowedHosts.includes(endpoint.hostname)
      ? endpoint.hostname
      : allowedHosts[0];
    if (announcedHost === undefined)
      throw new StreamableHttpHostAllowlistError(bound.address);
    endpoint.hostname = announcedHost;
    url = endpoint.href;
    origin = endpoint.origin;
  } catch (cause: unknown) {
    await prepareShutdown().catch(() => undefined);
    throw new StreamableHttpListenError(cause);
  }

  httpServer.on("request", (request, response) => {
    void requests.handle(request, response).catch((cause: unknown) => {
      logger?.warn(
        { error: errorMessage(cause) },
        "MCP HTTP request did not complete",
      );
      response.destroy();
    });
  });
  httpServer.on("error", (error) =>
    options.onerror?.(new StreamableHttpListenError(error, "serving")),
  );

  if (options.signal?.aborted) {
    await prepareShutdown();
    throw options.signal.reason;
  }
  logger
    ?.child({}, { level: "info" })
    .info({ transport: "streamable-http", url }, "MCP transport listening");
  return { url, prepareShutdown, close: prepareShutdown };
};

/** Stop accepting, drop open connections and release the port. */
const closeHttpServer = (
  httpServer: ReturnType<typeof createHttpServer>,
): Promise<void> => {
  const closed = new Promise<void>((resolve, reject) => {
    httpServer.close((error) =>
      error === undefined ? resolve() : reject(error),
    );
  });
  httpServer.closeAllConnections();
  return closed;
};

/** Translate one node:http exchange through `route`, logging a failure and answering it when headers are still open. */
const serveNodeRequest = async (
  target: NodeRequestTarget,
  request: IncomingMessage,
  response: ServerResponse,
  context: NodeHttpExchangeContext,
): Promise<void> => {
  const { route, origin, logger } = target;
  const declaresBody =
    request.headers["content-length"] !== undefined ||
    request.headers["transfer-encoding"] !== undefined;
  let webRequest: Request | undefined;
  const unreadBody = (): boolean => declaresBody && !request.readableEnded;
  let reported = false;
  let admissionLifetime: HttpAdmissionLifetime = "response";
  const settled = (): void => {
    if (reported) return;
    reported = true;
    context.onDispatchSettled({
      hasUnreadBody: unreadBody(),
      admissionLifetime,
    });
  };
  try {
    try {
      webRequest = toWebRequest(request, response, origin());
    } catch (cause: unknown) {
      const target = cause instanceof UnsupportedRequestTargetError;
      await writeWebResponse(
        jsonError(
          target ? 400 : 405,
          -32600,
          target
            ? "Unsupported request target"
            : `Unsupported method ${request.method ?? ""}`,
        ),
        response,
        { closeConnection: true, onBodyCleanup: context.onBodyCleanup },
      );
      return;
    }
    let webResponse: Response;
    try {
      const routed = await route(webRequest);
      webResponse = routed.response;
      admissionLifetime = routed.admissionLifetime;
    } finally {
      settled();
    }
    await writeWebResponse(webResponse, response, {
      closeConnection: unreadBody(),
      onBodyCleanup: context.onBodyCleanup,
    });
  } catch (cause: unknown) {
    logger?.[response.headersSent ? "debug" : "warn"](
      { error: errorMessage(cause) },
      "MCP HTTP request did not complete",
    );
    if (response.headersSent) response.destroy();
    else
      await writeWebResponse(
        jsonError(500, -32603, "Internal server error"),
        response,
        { closeConnection: unreadBody(), onBodyCleanup: context.onBodyCleanup },
      ).catch(() => undefined);
  } finally {
    settled();
  }
};

/** Run close stages in order once each; a failed stage is retried by the next call instead of replaying its rejection. */
const stagedClose = (
  stages: readonly (() => Promise<void>)[],
): (() => Promise<void>) => {
  const done = stages.map(() => false);
  let closing: Promise<void> | undefined;
  return () => {
    if (closing !== undefined) return closing;
    const attempt = (async () => {
      for (const [index, stage] of stages.entries()) {
        if (done[index]) continue;
        await stage();
        done[index] = true;
      }
    })();
    closing = attempt;
    void attempt.catch(() => {
      if (closing === attempt) closing = undefined;
    });
    return attempt;
  };
};
