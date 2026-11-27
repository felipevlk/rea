import { once } from "node:events";
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
import { Agent, request as httpRequest } from "node:http";
import { connect as connectTcp, createServer, type Socket } from "node:net";
import pino from "pino";
import { afterEach, expect, it, vi } from "vitest";
import { z } from "zod";

import {
  expectForeignHostRefused,
  rawHttpRequest,
} from "../../../tests/support/http/rawRequest.js";
import {
  phaseArrival,
  phaseGate,
} from "../../../tests/support/lifecycleHarness.js";
import { closeVerifierResources } from "../../../scripts/lib/verifier-cleanup.mjs";
import {
  hostHeaderAllowlist,
  serveStreamableHttp,
  type StreamableHttpServer,
} from "./serveStreamableHttp.js";
import { StreamableHttpListenError } from "./streamableHttpListenError.js";
import { mcpAllowedHostsTextSchema } from "../../config/mcpHttpEndpoint.js";
import { classifyCliEntrypoint } from "../../cliBuiltinArguments.js";
import { validateMcpHttpCommand } from "../../main/mcpHttpArguments.js";
import { RequestTrackingMcpServer } from "../RequestTrackingMcpServer.js";
import { testWaitDeadline, waitFor } from "../../../tests/support/testWait.js";
import { ProviderCleanupError } from "../../domain/providerCleanupError.js";

const servers: StreamableHttpServer[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((server) => server.close()));
});

/** An echo server whose protocol onclose hook counts as a shutdown, as rea's provider cleanup does. */
const createEchoServer = (onShutdown: () => void): RequestTrackingMcpServer => {
  const server = new RequestTrackingMcpServer({
    name: "echo-fixture",
    version: "1",
  });
  server.registerTool(
    "echo",
    { description: "Echo text", inputSchema: z.object({ text: z.string() }) },
    ({ text }) => ({ content: [{ type: "text", text }] }),
  );
  server.server.onclose = onShutdown;
  return server;
};

const start = async (
  overrides: Partial<Parameters<typeof serveStreamableHttp>[0]> = {},
) => {
  const logs: string[] = [];
  const logger = pino({ level: "debug" }, { write: (line) => logs.push(line) });
  let factoryCalls = 0;
  let shutdownHooks = 0;
  const served = await serveStreamableHttp({
    host: "127.0.0.1",
    port: 0,
    createMcpServer: () => {
      factoryCalls += 1;
      return createEchoServer(() => {
        shutdownHooks += 1;
      });
    },
    logger,
    ...overrides,
  });
  servers.push(served);
  return {
    served,
    messages: () =>
      logs.map((line) => (JSON.parse(line) as { msg: string }).msg),
    entries: () =>
      logs.map((line) => JSON.parse(line) as Record<string, unknown>),
    factoryCalls: () => factoryCalls,
    shutdownHooks: () => shutdownHooks,
  };
};

const connect = async (url: string) => {
  const transport = new StreamableHTTPClientTransport(new URL(url));
  const client = new Client(
    { name: "streamable-http-adapter", version: "1" },
    { versionNegotiation: { mode: { pin: "2026-07-28" } } },
  );
  await client.connect(transport);
  return { transport, client };
};

/** Reserve an ephemeral port, release it, and return the number for a later bind. */
const listenEphemeral = async (): Promise<number> =>
  new Promise((resolve, reject) => {
    const probe = createServer();
    probe.unref();
    probe.once("listening", () => {
      const { port } = probe.address() as { port: number };
      probe.close(() =>
        port > 0 ? resolve(port) : reject(new Error("no port")),
      );
    });
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1");
  });

const echo = async (client: Client, text: string): Promise<unknown> => {
  const result = await client.callTool({ name: "echo", arguments: { text } });
  return result.content;
};

/** Return the socket that actually carried a complete reply. */
const keepAliveRequest = (
  agent: Agent,
  url: URL,
  options: {
    readonly method: string;
    readonly headers?: Readonly<Record<string, string>>;
    readonly body?: string;
  },
): Promise<{
  status: number;
  headers: Record<string, string | string[] | undefined>;
  socket: Socket;
}> =>
  new Promise((resolve, reject) => {
    const request = httpRequest(
      url,
      { method: options.method, headers: { ...options.headers }, agent },
      (response) => {
        const socket = response.socket;
        response.resume();
        response.once("end", () =>
          resolve({
            status: response.statusCode ?? 0,
            headers: response.headers,
            socket,
          }),
        );
        response.once("error", reject);
      },
    );
    request.once("error", reject);
    request.end(options.body);
  });

const waitForSocketClose = async (socket: Socket) => {
  if (!socket.destroyed)
    await once(socket, "close", {
      signal: AbortSignal.timeout(testWaitDeadline()),
    });
};

const MODERN_VERSION = "2026-07-28";

/** One 2026-07-28 JSON-RPC request body with its per-request `_meta` envelope. */
const modernBody = (method: string, params: Record<string, unknown> = {}) =>
  JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method,
    params: {
      ...params,
      _meta: {
        "io.modelcontextprotocol/protocolVersion": MODERN_VERSION,
        "io.modelcontextprotocol/clientInfo": { name: "test", version: "1" },
        "io.modelcontextprotocol/clientCapabilities": {},
      },
    },
  });

const modernHeaders = (method: string, extra: Record<string, string> = {}) => ({
  "content-type": "application/json",
  accept: "application/json, text/event-stream",
  "mcp-protocol-version": MODERN_VERSION,
  "mcp-method": method,
  ...extra,
});

/** A raw 2026 request that bypasses the SDK client, so the wire shape is explicit. */
const modernRequest = (
  url: URL,
  method: string,
  params: Record<string, unknown> = {},
  extra: Record<string, string> = {},
) =>
  rawHttpRequest(url, {
    method: "POST",
    headers: modernHeaders(method, extra),
    body: modernBody(method, params),
  });

/** A legacy 2025 initialize, which a modern-only endpoint must refuse. */
const legacyInitialize = () =>
  JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-11-25",
      capabilities: {},
      clientInfo: { name: "legacy-client", version: "1" },
    },
  });

it("announces the bound URL and serves stateless 2026-07-28 calls to concurrent clients", async () => {
  const { served, entries, factoryCalls } = await start();
  expect(new URL(served.url).pathname).toBe("/mcp");
  expect(entries()).toContainEqual(
    expect.objectContaining({
      transport: "streamable-http",
      url: served.url,
      msg: "MCP transport listening",
    }),
  );
  const first = await connect(served.url);
  const second = await connect(served.url);
  // The modern era has no MCP session id; each request is independent.
  expect(first.transport.sessionId).toBeUndefined();
  expect(second.transport.sessionId).toBeUndefined();
  await expect(echo(first.client, "first")).resolves.toEqual([
    { type: "text", text: "first" },
  ]);
  await expect(echo(second.client, "second")).resolves.toEqual([
    { type: "text", text: "second" },
  ]);
  // The endpoint owns one instance per request and hands each to the factory.
  expect(factoryCalls()).toBeGreaterThanOrEqual(2);
  await first.client.close();
  await second.client.close();
});

it("answers server/discover with 2026-07-28 and the echo server's capabilities", async () => {
  const { served } = await start();
  const discovered = await modernRequest(
    new URL(served.url),
    "server/discover",
  );
  expect(discovered.status).toBe(200);
  expect(JSON.parse(discovered.body)).toMatchObject({
    result: {
      supportedVersions: [MODERN_VERSION],
      capabilities: { tools: { listChanged: true } },
    },
  });
});

it("rejects the 2025 initialize handshake with the modern-only supported list", async () => {
  const { served } = await start();
  const refused = await rawHttpRequest(new URL(served.url), {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
    },
    body: legacyInitialize(),
  });
  expect(refused.status).toBe(400);
  expect(JSON.parse(refused.body)).toMatchObject({
    jsonrpc: "2.0",
    error: {
      code: -32022,
      message: expect.stringContaining("2025-11-25"),
      data: { supported: [MODERN_VERSION], requested: "2025-11-25" },
    },
  });
});

it("refuses 2025 session operations instead of opening a stream or session", async () => {
  const { served } = await start();
  const url = new URL(served.url);
  for (const method of ["GET", "DELETE"]) {
    const refused = await rawHttpRequest(url, {
      method,
      headers: {
        accept: "text/event-stream",
        "mcp-session-id": "legacy-session",
        "mcp-protocol-version": "2025-11-25",
      },
    });
    expect(refused.status, method).toBeGreaterThanOrEqual(400);
    expect(refused.body, method).toContain('"jsonrpc":"2.0"');
  }
});

it("refuses batches, which the 2026 request envelope does not admit", async () => {
  const { served } = await start();
  const refused = await rawHttpRequest(new URL(served.url), {
    method: "POST",
    headers: modernHeaders("server/discover"),
    body: `[${modernBody("server/discover")}]`,
  });
  expect(refused.status).toBe(400);
  expect(JSON.parse(refused.body)).toMatchObject({
    jsonrpc: "2.0",
    error: { code: expect.any(Number) },
  });
});

it("serves plain routes beside the MCP endpoint and answers unknown paths with 404", async () => {
  const { served } = await start({
    routes: [
      {
        method: "GET",
        path: "/.well-known/probe",
        handle: () => Response.json({ probe: true }),
      },
    ],
  });
  const probe = await fetch(new URL("/.well-known/probe", served.url));
  await expect(probe.json()).resolves.toEqual({ probe: true });
  const missing = await fetch(new URL("/missing", served.url));
  expect(missing.status).toBe(404);
  await expect(missing.json()).resolves.toMatchObject({
    error: { code: -32601 },
  });
  const wrongMethod = await fetch(new URL("/.well-known/probe", served.url), {
    method: "POST",
    body: "{}",
  });
  expect(wrongMethod.status).toBe(404);
});

it("refuses foreign Host headers on a loopback bind and frees the port on close", async () => {
  const { served } = await start();
  const url = new URL(served.url);
  await expectForeignHostRefused(url);
  await served.close();
  await served.close();
  await expect(rawHttpRequest(url)).rejects.toMatchObject({
    code: "ECONNREFUSED",
  });
});

it("refuses a non-loopback bind without an operator host allowlist as a typed listen error and frees the port", async () => {
  const port = await listenEphemeral();
  const refused = start({ host: "0.0.0.0", port });
  await expect(refused).rejects.toBeInstanceOf(StreamableHttpListenError);
  await expect(refused).rejects.toThrow(/MCP_ALLOWED_HOSTS/u);
  // The refusal happens after the bind, so the staged close must have released it.
  // `start` owns the re-bound listener: afterEach closes it even if an assertion above throws.
  const rebind = start({ host: "127.0.0.1", port });
  await expect(rebind).resolves.toMatchObject({
    served: { close: expect.any(Function) },
  });
});

it.each([
  ["REA.LAN", "rea.lan"],
  ["rea.lan.", "rea.lan."],
  ["bücher.example", "xn--bcher-kva.example"],
  ["127.0.0.1", "127.0.0.1"],
  ["::1", "[::1]"],
  ["0:0:0:0:0:0:0:1", "[::1]"],
  ["[0:0:0:0:0:0:0:1]", "[::1]"],
])("preserves declared hostname %j as %j", async (name, normalized) => {
  const { served } = await start({
    host: "0.0.0.0",
    allowedHosts: mcpAllowedHostsTextSchema.parse(name),
    routes: [
      {
        method: "GET",
        path: "/probe",
        handle: () => Response.json({ allowed: true }),
      },
    ],
  });
  const url = new URL(served.url);
  url.hostname = "127.0.0.1";
  url.pathname = "/probe";
  const response = await rawHttpRequest(url, {
    headers: {
      host: `${normalized}:${url.port}`,
      origin: `http://${normalized}:${url.port}`,
    },
  });
  expect(response.status).toBe(200);
  expect(JSON.parse(response.body)).toEqual({ allowed: true });
});

it("serves every declared repeated host while refusing undeclared hosts and origins", async () => {
  const entry = classifyCliEntrypoint(
    [
      "mcp",
      "--http",
      "--allowed-hosts",
      "a.example",
      "--allowed-hosts=b.example",
    ],
    {},
  );
  if (entry.kind !== "mcp-http") throw new Error("Expected an MCP HTTP entry");
  const parsed = validateMcpHttpCommand(entry.command);
  if (!parsed.ok || parsed.allowedHosts === undefined)
    throw new Error("Expected valid repeated host lists");
  const { served } = await start({
    host: "0.0.0.0",
    allowedHosts: parsed.allowedHosts,
    routes: [
      {
        method: "GET",
        path: "/probe",
        handle: () => Response.json({ allowed: true }),
      },
    ],
  });
  const url = new URL(served.url);
  url.hostname = "127.0.0.1";
  url.pathname = "/probe";
  for (const host of ["a.example", "b.example"]) {
    const response = await rawHttpRequest(url, {
      headers: { host, origin: `http://${host}` },
    });
    expect(response.status, host).toBe(200);
    expect(JSON.parse(response.body)).toEqual({ allowed: true });
  }
  expect(
    (await rawHttpRequest(url, { headers: { host: "undeclared.example" } }))
      .status,
  ).toBe(403);
  expect(
    (
      await rawHttpRequest(url, {
        headers: { host: "a.example", origin: "http://undeclared.example" },
      })
    ).status,
  ).toBe(403);
});

it("announces an admitted non-loopback URL and gates it on the operator allowlist", async () => {
  const { served } = await start({
    host: "0.0.0.0",
    allowedHosts: mcpAllowedHostsTextSchema.parse("127.0.0.1"),
  });
  const url = new URL(served.url);
  expect(url.hostname).toBe("127.0.0.1");
  const { client } = await connect(url.toString());
  await expect(echo(client, "allowed")).resolves.toEqual([
    { type: "text", text: "allowed" },
  ]);
  await client.close();
  await expectForeignHostRefused(url);
});

it("reports a port that cannot be bound as a typed listen error naming the cause", async () => {
  const { served } = await start();
  const failed = serveStreamableHttp({
    host: "127.0.0.1",
    port: Number(new URL(served.url).port),
    createMcpServer: () => createEchoServer(() => undefined),
  });
  await expect(failed).rejects.toBeInstanceOf(StreamableHttpListenError);
  await expect(failed).rejects.toThrow(/EADDRINUSE/u);
});

it("closes a listener whose startup signal aborted before it bound and frees the port", async () => {
  const port = await listenEphemeral();
  const reason = new Error("startup cancelled");
  const controller = new AbortController();
  controller.abort(reason);
  await expect(
    serveStreamableHttp({
      host: "127.0.0.1",
      port,
      signal: controller.signal,
      createMcpServer: () => createEchoServer(() => undefined),
    }),
  ).rejects.toBe(reason);
  const rebind = await start({ port });
  expect(rebind.served.close).toBeTypeOf("function");
});

it("keeps the SDK loopback allowlist, plus the bind's own hostname, for every loopback spelling", () => {
  const sdk = ["localhost", "127.0.0.1", "[::1]"];
  expect(hostHeaderAllowlist("127.0.0.1", "127.0.0.1")).toEqual(sdk);
  expect(hostHeaderAllowlist("localhost", "127.0.0.1")).toEqual(sdk);
  expect(hostHeaderAllowlist("LOCALHOST", "::1")).toEqual(sdk);
  expect(hostHeaderAllowlist("::1", "::1")).toEqual(sdk);
  expect(hostHeaderAllowlist("0:0:0:0:0:0:0:1", "::1")).toEqual(sdk);
  expect(hostHeaderAllowlist("127.0.0.2", "127.0.0.2")).toEqual([
    ...sdk,
    "127.0.0.2",
  ]);
  expect(hostHeaderAllowlist("localhost.", "127.0.0.1")).toEqual([
    ...sdk,
    "localhost.",
  ]);
  expect(hostHeaderAllowlist("::ffff:127.0.0.1", "::ffff:127.0.0.1")).toEqual([
    ...sdk,
    "[::ffff:7f00:1]",
  ]);
  expect(hostHeaderAllowlist("192.0.2.10", "192.0.2.10")).toBeUndefined();
  expect(hostHeaderAllowlist("0.0.0.0", "0.0.0.0")).toBeUndefined();
  expect(hostHeaderAllowlist("::", "::")).toBeUndefined();
  expect(hostHeaderAllowlist("example.com", "192.0.2.10")).toBeUndefined();
  expect(hostHeaderAllowlist("not a host", "192.0.2.10")).toBeUndefined();
});

it("accepts a same-origin browser Origin carrying the bound port on the MCP endpoint and on routes", async () => {
  const { served } = await start({
    routes: [
      { method: "GET", path: "/health", handle: () => new Response("ok") },
    ],
  });
  const url = new URL(served.url);
  for (const name of ["127.0.0.1", "localhost"]) {
    const headers = {
      host: `${name}:${url.port}`,
      origin: `http://${name}:${url.port}`,
    };
    const route = await rawHttpRequest(new URL("/health", url), { headers });
    expect(route.status).toBe(200);
    const discovered = await modernRequest(
      url,
      "server/discover",
      {},
      { origin: headers.origin, host: headers.host },
    );
    expect(discovered.status).toBe(200);
  }
});

it("announces the bound URL even when the log level hides info lines", async () => {
  const logs: string[] = [];
  const logger = pino({ level: "warn" }, { write: (line) => logs.push(line) });
  const { served } = await start({ logger });
  expect(
    logs.map((line) => JSON.parse(line) as Record<string, unknown>),
  ).toContainEqual(
    expect.objectContaining({
      msg: "MCP transport listening",
      url: served.url,
    }),
  );
});

it("classifies loopback from the bound address and lists the configured name beside it", () => {
  const sdk = ["localhost", "127.0.0.1", "[::1]"];
  expect(hostHeaderAllowlist("ip6-localhost", "::1")).toEqual([
    ...sdk,
    "ip6-localhost",
  ]);
  expect(hostHeaderAllowlist("dev.local", "127.0.0.1")).toEqual([
    ...sdk,
    "dev.local",
  ]);
  expect(hostHeaderAllowlist("0.0.0.0", "0.0.0.0")).toBeUndefined();
  expect(hostHeaderAllowlist("dev.local", "192.0.2.10")).toBeUndefined();
});

it.each([
  ["localhost", "192.0.2.10"],
  ["service.localhost", "2001:db8::10"],
])(
  "does not classify configured %s as loopback when bound to %s",
  (host, address) => {
    expect(hostHeaderAllowlist(host, address)).toBeUndefined();
  },
);

it("refuses a scoped IPv6 bind before creating anything and keeps bind failures away from onerror", async () => {
  let factoryCalls = 0;
  const onerror = (): void => {
    throw new Error("onerror must not run during startup");
  };
  await expect(
    serveStreamableHttp({
      host: "fe80::1%lo0",
      port: 0,
      onerror,
      createMcpServer: () => {
        factoryCalls += 1;
        return createEchoServer(() => undefined);
      },
    }),
  ).rejects.toBeInstanceOf(StreamableHttpListenError);
  expect(factoryCalls).toBe(0);
  const { served } = await start();
  await expect(
    serveStreamableHttp({
      host: "127.0.0.1",
      port: Number(new URL(served.url).port),
      onerror,
      createMcpServer: () => createEchoServer(() => undefined),
    }),
  ).rejects.toThrow(/EADDRINUSE/u);
});

it("answers an oversized modern POST promptly instead of stalling on the probe", async () => {
  const { served } = await start();
  const { client } = await connect(served.url);
  const url = new URL(served.url);
  // The SDK refuses the body as soon as it exceeds the bound, so the upload may end in a broken pipe.
  await rawHttpRequest(url, {
    method: "POST",
    headers: modernHeaders("server/discover"),
    body: `{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"pad":"${"x".repeat(4_500_000)}","_meta":{"io.modelcontextprotocol/protocolVersion":"${MODERN_VERSION}"}}}`,
  }).then(
    (refused) => {
      expect(refused.status).toBeGreaterThanOrEqual(400);
      expect(refused.status).toBeLessThan(500);
    },
    (cause: unknown) => {
      expect(["EPIPE", "ECONNRESET"]).toContain(
        (cause as { code?: string }).code,
      );
    },
  );
  expect(await echo(client, "still served")).toEqual([
    { type: "text", text: "still served" },
  ]);
  await client.close();
});

it("closes the connection after a reply sent before the request body was read, on a keep-alive client", async () => {
  const { served } = await start();
  const url = new URL(served.url);
  const agent = new Agent({ keepAlive: true, maxSockets: 1 });
  try {
    const unread = await keepAliveRequest(agent, new URL("/nowhere", url), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: '{"never":"read"}',
    });
    expect(unread.status).toBe(404);
    expect(unread.headers.connection).toBe("close");
    await waitForSocketClose(unread.socket);
    // Node's parser refuses a chunked GET body itself; a declared Content-Length reaches the bridge.
    const bodyOnGet = await keepAliveRequest(agent, new URL("/nowhere", url), {
      method: "GET",
      headers: { "content-type": "text/plain", "content-length": "15" },
      body: "a body on a GET",
    });
    expect(bodyOnGet.status).toBe(404);
    expect(bodyOnGet.headers.connection).toBe("close");
    await waitForSocketClose(bodyOnGet.socket);
    const bodiless = await keepAliveRequest(agent, new URL("/nowhere", url), {
      method: "GET",
    });
    expect(bodiless.status).toBe(404);
    expect(bodiless.headers.connection).not.toBe("close");
    const repeated = await keepAliveRequest(agent, new URL("/nowhere", url), {
      method: "GET",
    });
    expect(repeated.status).toBe(404);
    expect(repeated.socket).toBe(bodiless.socket);
  } finally {
    agent.destroy();
  }
});

it("answers a method Fetch refuses instead of letting it escape the request guard", async () => {
  const { served } = await start();
  const url = new URL(served.url);
  const trace = await rawHttpRequest(url, {
    method: "TRACE",
    headers: { host: `127.0.0.1:${url.port}` },
  });
  expect(trace.status).toBe(405);
  expect(trace.headers.connection).toBe("close");
  const { client } = await connect(served.url);
  expect(await echo(client, "still alive")).toEqual([
    { type: "text", text: "still alive" },
  ]);
  await client.close();
});

/** One request with a caller-controlled request target, which URL-based clients normalize away. */
const rawTargetRequest = (
  served: URL,
  target: string,
): Promise<{ status: number; body: string }> =>
  new Promise((resolve, reject) => {
    const request = httpRequest(
      {
        host: served.hostname,
        port: Number(served.port),
        path: target,
        method: "GET",
        agent: false,
        headers: { host: `127.0.0.1:${served.port}` },
      },
      (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk: string) => {
          body += chunk;
        });
        response.on("end", () =>
          resolve({ status: response.statusCode ?? 0, body }),
        );
      },
    );
    request.on("error", reject);
    request.end();
  });

it("keeps a network-path request target as a path and refuses absolute-form targets", async () => {
  const seen: string[] = [];
  const { served } = await start({
    routes: [
      {
        method: "GET",
        path: "/card",
        handle: (request) => {
          seen.push(new URL(request.url).host);
          return Response.json({ ok: true });
        },
      },
    ],
  });
  const url = new URL(served.url);
  const networkPath = await rawTargetRequest(url, "//attacker.example/card");
  expect(networkPath.status).toBe(404);
  expect(networkPath.body).toContain(
    "No route for GET //attacker.example/card",
  );
  const absolute = await rawTargetRequest(url, "http://attacker.example/card");
  expect(absolute.status).toBe(400);
  // Dot segments would route a different path than the one requested; they are refused, not canonicalized.
  for (const target of ["/public/../mcp", "/public/%2e%2e/mcp", "/./mcp"])
    expect((await rawTargetRequest(url, target)).status).toBe(400);
  expect((await rawTargetRequest(url, "/card?x=1")).status).toBe(200);
  const plain = await rawTargetRequest(url, "/card");
  expect(plain.status).toBe(200);
  // Both successful route calls saw only the bound host, never the attacker-selected one.
  expect(seen).toEqual([`127.0.0.1:${url.port}`, `127.0.0.1:${url.port}`]);
});

it("refuses raw fragments without changing an encoded query value", async () => {
  const handled = vi.fn((request: Request) =>
    Response.json({
      token: new URL(request.url).searchParams.get("token"),
      url: request.url,
    }),
  );
  const { served } = await start({
    routes: [{ method: "GET", path: "/plain", handle: handled }],
  });
  const url = new URL(served.url);
  for (const target of ["/plain?token=a#b", "/plain?token=a#"])
    expect((await rawTargetRequest(url, target)).status).toBe(400);
  expect(handled).not.toHaveBeenCalled();
  const valid = await rawTargetRequest(url, "/plain?token=a%23b");
  expect(valid.status).toBe(200);
  expect(JSON.parse(valid.body)).toEqual({
    token: "a#b",
    url: `${url.origin}/plain?token=a%23b`,
  });
});

it("closes the connection when a route read only part of the body and keeps it after a full read", async () => {
  const { served } = await start({
    routes: [
      {
        method: "POST",
        path: "/partial",
        handle: async (request) => {
          const reader = request.body?.getReader();
          await reader?.read();
          return new Response("stopped early");
        },
      },
      {
        method: "POST",
        path: "/whole",
        handle: async (request) =>
          Response.json({ bytes: (await request.text()).length }),
      },
    ],
  });
  const url = new URL(served.url);
  const agent = new Agent({ keepAlive: true, maxSockets: 1 });
  const body = "x".repeat(256 * 1024);
  try {
    const partial = await keepAliveRequest(agent, new URL("/partial", url), {
      method: "POST",
      headers: {
        "content-type": "text/plain",
        "content-length": String(body.length),
      },
      body,
    });
    expect(partial.status).toBe(200);
    expect(partial.headers.connection).toBe("close");
    await waitForSocketClose(partial.socket);
    const whole = await keepAliveRequest(agent, new URL("/whole", url), {
      method: "POST",
      headers: {
        "content-type": "text/plain",
        "content-length": String(body.length),
      },
      body,
    });
    expect(whole.status).toBe(200);
    expect(whole.headers.connection).not.toBe("close");
    const repeated = await keepAliveRequest(agent, new URL("/whole", url), {
      method: "POST",
      headers: { "content-length": "1" },
      body: "x",
    });
    expect(repeated.status).toBe(200);
    expect(repeated.socket).toBe(whole.socket);
  } finally {
    agent.destroy();
  }
});

it.each(["open stream", "late response"])(
  "joins response-body cancellation before shutting down for %s",
  async (timing) => {
    const routeStart = phaseArrival();
    const bodyCancel = phaseGate();
    const { served } = await start({
      routes: [
        {
          method: "GET",
          path: "/stream",
          handle: async (request) => {
            routeStart.arrive();
            if (timing === "late response")
              await new Promise<void>((resolve) => {
                request.signal.addEventListener("abort", () => resolve(), {
                  once: true,
                });
              });
            return new Response(
              new ReadableStream<Uint8Array>({
                start(controller) {
                  controller.enqueue(
                    new TextEncoder().encode("data: ready\n\n"),
                  );
                },
                async cancel() {
                  await bodyCancel.hold();
                },
              }),
            );
          },
        },
      ],
    });
    const response = fetch(new URL("/stream", served.url)).catch(
      () => undefined,
    );
    let closing: Promise<void> | undefined;
    try {
      await routeStart.reached(response, "route start");
      if (timing === "open stream") await response;
      let closed = false;
      closing = served.prepareShutdown().then(() => {
        closed = true;
      });
      await bodyCancel.reached(closing, "body cancellation");
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(closed).toBe(false);
      bodyCancel.release();
      await closing;
    } finally {
      bodyCancel.release();
      await closeVerifierResources([
        async () => {
          await (closing ?? served.close());
        },
        async () => {
          await (await response)?.body?.cancel().catch(() => undefined);
        },
      ]);
    }
  },
);

it.each(["chunk", "headers"])(
  "joins response cleanup after an output failure in %s",
  async (failure) => {
    const responseCleanup = phaseGate();
    let cancelled = 0;
    const source = new ReadableStream<Uint8Array>({
      start(controller) {
        // Node's Response accepts this malformed producer; the HTTP writer rejects it.
        controller.enqueue(
          failure === "chunk"
            ? ({} as unknown as Uint8Array)
            : new TextEncoder().encode("pending"),
        );
      },
      async cancel() {
        cancelled += 1;
        await responseCleanup.hold();
      },
    });
    const { served } = await start({
      maxConcurrentDispatches: 1,
      routes: [
        { method: "GET", path: "/health", handle: () => new Response("ok") },
        {
          method: "GET",
          path: "/broken",
          handle: () =>
            new Response(
              source,
              failure === "headers"
                ? { headers: { "x-invalid": "\u0001" } }
                : undefined,
            ),
        },
      ],
    });
    const received = fetch(new URL("/broken", served.url))
      .then(async (response) => {
        await response.arrayBuffer();
      })
      .catch(() => undefined);
    let closing: Promise<void> | undefined;
    try {
      await waitFor(() => {
        expect(cancelled).toBe(1);
      });
      expect((await fetch(new URL("/health", served.url))).status).toBe(503);
      let closed = false;
      closing = served.prepareShutdown().then(() => {
        closed = true;
      });
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(closed).toBe(false);
      responseCleanup.release();
      await closing;
      expect(cancelled).toBe(1);
    } finally {
      responseCleanup.release();
      await source.cancel().catch(() => undefined);
      await closing;
      await received;
    }
  },
);

it("keeps HEAD delivery prompt while joining a retained tee's cleanup on shutdown", async () => {
  let sourceClosed = false;
  const original = new Response(
    new ReadableStream<Uint8Array>({
      pull: () => new Promise(() => undefined),
      cancel: () => {
        sourceClosed = true;
      },
    }),
  );
  const { served } = await start({
    routes: [
      { method: "HEAD", path: "/cached", handle: () => original.clone() },
    ],
  });
  const response = await fetch(new URL("/cached", served.url), {
    method: "HEAD",
  });
  expect(response.status).toBe(200);
  let closed = false;
  const closing = served.prepareShutdown().then(() => {
    closed = true;
  });
  try {
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(closed).toBe(false);
    expect(sourceClosed).toBe(false);
    await original.body?.cancel();
    await closing;
    expect(sourceClosed).toBe(true);
  } finally {
    await original.body?.cancel();
    await closing;
  }
});

it("retains failed response cleanup and blocks provider teardown and further admission", async () => {
  const failure = new Error("response finalizer failed");
  const { served } = await start({
    routes: [
      { method: "GET", path: "/health", handle: () => new Response("ok") },
      {
        method: "GET",
        path: "/failed",
        handle: () =>
          new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                controller.enqueue(new TextEncoder().encode("ready"));
              },
              cancel: () => Promise.reject(failure),
            }),
          ),
      },
    ],
  });
  const response = await fetch(new URL("/failed", served.url));
  try {
    await response.body?.cancel();
    await waitFor(async () => {
      expect((await fetch(new URL("/health", served.url))).status).toBe(503);
    });
    await expect(served.prepareShutdown()).rejects.toMatchObject({
      _tag: "ProviderAdapterError",
      cleanupIncomplete: true,
      cleanupResources: ["MCP HTTP response source"],
      diagnostics: { failed_cleanup_count: 1 },
      cause: expect.any(AggregateError),
    });
    await expect(served.close()).rejects.toBeInstanceOf(ProviderCleanupError);
    await expect(rawHttpRequest(new URL(served.url))).rejects.toMatchObject({
      code: "ECONNREFUSED",
    });
  } finally {
    await served.close().catch(() => undefined);
    // The deliberate finalizer failure is retained; this fixture owns no real provider.
    servers.splice(servers.indexOf(served), 1);
  }
});

it("retains plain-response admission until asynchronous cancellation settles", async () => {
  const streamCancel = phaseGate();
  const health = vi.fn(() => new Response("ok"));
  const { served } = await start({
    maxConcurrentDispatches: 1,
    routes: [
      { method: "GET", path: "/health", handle: health },
      {
        method: "GET",
        path: "/stream",
        handle: () =>
          new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                controller.enqueue(new TextEncoder().encode("ready"));
              },
              async cancel() {
                await streamCancel.hold();
              },
            }),
          ),
      },
    ],
  });
  const response = await fetch(new URL("/stream", served.url));
  try {
    expect((await fetch(new URL("/health", served.url))).status).toBe(503);
    await response.body?.cancel();
    // The client-side cancel has already settled; only the deadline bounds this wait.
    await streamCancel.reached(
      new Promise(() => {}),
      "server stream cancellation",
    );
    expect((await fetch(new URL("/health", served.url))).status).toBe(503);
    expect(health).not.toHaveBeenCalled();
    streamCancel.release();
    await waitFor(async () => {
      expect((await fetch(new URL("/health", served.url))).status).toBe(200);
    });
  } finally {
    streamCancel.release();
    await response.body?.cancel();
  }
});

it.each(["pending handler", "open response"])(
  "closes a peer pipelining behind a %s without queuing refusals",
  async (mode) => {
    const firstRoute = phaseArrival();
    const fallback = phaseGate();
    let finalized = 0;
    let secondCalled = 0;
    const held = async (request: Request) => {
      try {
        await Promise.race([
          fallback.hold(),
          new Promise<void>((resolve) => {
            request.signal.addEventListener("abort", () => resolve(), {
              once: true,
            });
          }),
        ]);
      } finally {
        finalized += 1;
      }
      return new Response("done");
    };
    const health = vi.fn(() => new Response("ok"));
    const { served } = await start({
      maxConcurrentDispatches: 2,
      routes: [
        {
          method: "GET",
          path: "/first",
          handle: (request) => {
            firstRoute.arrive();
            return mode === "pending handler"
              ? held(request)
              : new Response(
                  new ReadableStream<Uint8Array>({
                    start(controller) {
                      controller.enqueue(
                        new TextEncoder().encode("data: ready\n\n"),
                      );
                    },
                  }),
                );
          },
        },
        {
          method: "GET",
          path: "/second",
          handle: (request) => {
            secondCalled += 1;
            return held(request);
          },
        },
        { method: "GET", path: "/health", handle: health },
      ],
    });
    const url = new URL(served.url);
    const socket = connectTcp(Number(url.port), url.hostname);
    socket.on("error", () => undefined);
    socket.on("data", () => undefined);
    const request = (path: string) =>
      `GET ${path} HTTP/1.1\r\nHost: ${url.host}\r\n\r\n`;
    try {
      await new Promise<void>((resolve) => {
        socket.once("connect", resolve);
      });
      const socketClosed = new Promise<void>((resolve) => {
        socket.once("close", () => resolve());
      });
      socket.write(request("/first"));
      await firstRoute.reached(socketClosed, "first pipelined route");
      socket.write(request("/second"));
      socket.write(request("/health").repeat(32));
      await waitFor(() => {
        expect(socket.destroyed).toBe(true);
      });
      await waitFor(() => {
        expect(finalized).toBe(mode === "pending handler" ? 1 : 0);
      });
      expect(secondCalled).toBe(0);
      expect(health).not.toHaveBeenCalled();
    } finally {
      fallback.release();
      socket.destroy();
    }
  },
);

it.each(["prepareShutdown", "close"] as const)(
  "waits for plain-route finalizers before shutting down during %s",
  async (method) => {
    const routeStart = phaseArrival();
    const routeFinalizer = phaseGate();
    const { served } = await start({
      routes: [
        {
          method: "GET",
          path: "/pending",
          handle: async (request) => {
            routeStart.arrive();
            try {
              await new Promise<void>((resolve) => {
                request.signal.addEventListener("abort", () => resolve(), {
                  once: true,
                });
              });
            } finally {
              await routeFinalizer.hold();
            }
            return new Response("done");
          },
        },
      ],
    });
    const response = fetch(new URL("/pending", served.url)).catch(
      () => undefined,
    );
    let closing: Promise<void> | undefined;
    try {
      await routeStart.reached(response, "route start");
      let closed = false;
      closing = served[method]().then(() => {
        closed = true;
      });
      await routeFinalizer.reached(closing, "route finalizer");
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(closed).toBe(false);
      routeFinalizer.release();
      await closing;
    } finally {
      routeFinalizer.release();
      await closeVerifierResources([
        async () => {
          await (closing ?? served.close());
        },
        async () => {
          await response;
        },
      ]);
    }
  },
);

it("retains admission for an unread upload attached to a long response", async () => {
  const { served } = await start({
    maxConcurrentDispatches: 1,
    routes: [
      { method: "GET", path: "/health", handle: () => new Response("ok") },
      {
        method: "POST",
        path: "/stream",
        handle: async (request) => {
          const reader = request.body?.getReader();
          await reader?.read();
          reader?.releaseLock();
          return new Response(
            new ReadableStream<Uint8Array>({
              start(controller) {
                controller.enqueue(new TextEncoder().encode("data: ready\n\n"));
              },
            }),
            { headers: { "content-type": "text/event-stream" } },
          );
        },
      },
    ],
  });
  let received = false;
  const upload = httpRequest(
    new URL("/stream", served.url),
    {
      method: "POST",
      headers: { "content-length": "1024" },
    },
    (response) => {
      response.once("data", () => {
        received = true;
      });
      response.once("error", () => undefined);
      response.resume();
    },
  );
  upload.once("error", () => undefined);
  upload.write("prefix");
  try {
    await waitFor(() => {
      expect(received).toBe(true);
    });
    expect((await fetch(new URL("/health", served.url))).status).toBe(503);
    upload.destroy();
    await waitFor(async () => {
      expect((await fetch(new URL("/health", served.url))).status).toBe(200);
    });
  } finally {
    upload.destroy();
  }
});

it("bounds HTTP input dispatch globally and releases admission when an upload leaves", async () => {
  const { served } = await start({
    maxConcurrentDispatches: 2,
    routes: [
      { method: "GET", path: "/health", handle: () => new Response("ok") },
      {
        method: "POST",
        path: "/upload",
        handle: async (request) =>
          Response.json({ bytes: (await request.text()).length }),
      },
    ],
  });
  const first = await connect(served.url);
  const uploads: ReturnType<typeof httpRequest>[] = [];
  const pending: Promise<void>[] = [];
  try {
    for (let i = 0; i < 2; i += 1) {
      pending.push(
        new Promise<void>((resolve) => {
          const upload = httpRequest(
            new URL("/upload", served.url),
            {
              method: "POST",
              headers: { "content-type": "text/plain" },
            },
            (response) => {
              response.resume();
              response.once("end", resolve);
              response.once("error", () => resolve());
            },
          );
          uploads.push(upload);
          upload.once("error", () => resolve());
          upload.write("x".repeat(1024));
        }),
      );
    }
    // A plain route observes the same budget while both uploads wait for the rest of their bodies.
    await waitFor(async () => {
      expect((await fetch(new URL("/health", served.url))).status).toBe(503);
    });
    const refused = await rawHttpRequest(new URL(served.url), {
      method: "POST",
      headers: modernHeaders("server/discover"),
    });
    expect(refused.status).toBe(503);
    expect(refused.headers.connection).toBe("close");
    uploads[0]?.destroy();
    await pending[0];
    await waitFor(async () => {
      expect((await fetch(new URL("/health", served.url))).status).toBe(200);
    });
    expect(await echo(first.client, "kept during overload")).toEqual([
      { type: "text", text: "kept during overload" },
    ]);
  } finally {
    for (const upload of uploads) upload.destroy();
    await Promise.all(pending);
    await first.client.close();
  }
});

it("serves a JSON response when the client does not accept event streams", async () => {
  const { served } = await start();
  const discovered = await modernRequest(
    new URL(served.url),
    "server/discover",
    {},
    { accept: "application/json" },
  );
  expect(discovered.status).toBe(200);
  expect(discovered.headers["content-type"]).toContain("application/json");
  expect(JSON.parse(discovered.body)).toMatchObject({
    result: { supportedVersions: [MODERN_VERSION] },
  });
});

it.each([
  { status: 415, headers: { "content-type": "text/plain" } },
  { status: 413, headers: { "content-length": String(8 * 1024 * 1024) } },
])(
  "releases an unread body when the SDK refuses it with $status, keeping the endpoint live",
  async ({ status, headers }) => {
    const { served } = await start();
    const url = new URL(served.url);
    const refused = await rawHttpRequest(url, {
      method: "POST",
      headers: { ...modernHeaders("server/discover"), ...headers },
      ...(status === 413 ? {} : { body: modernBody("server/discover") }),
    });
    expect(refused.status).toBe(status);
    const { client } = await connect(served.url);
    expect(await echo(client, "still live")).toEqual([
      { type: "text", text: "still live" },
    ]);
    await client.close();
  },
);

it("ends a real HTTP request with an unfinished upload during shutdown", async () => {
  const { served } = await start();
  const first = await connect(served.url);
  let upload!: ReturnType<typeof httpRequest>;
  const response = new Promise<number | undefined>((resolve, reject) => {
    upload = httpRequest(
      served.url,
      {
        method: "POST",
        headers: modernHeaders("tools/call", { "mcp-name": "echo" }),
      },
      (reply) => {
        reply.resume();
        reply.once("end", () => resolve(reply.statusCode));
        reply.once("error", reject);
      },
    );
    upload.once("error", () => resolve(undefined));
    upload.write('{"jsonrpc":"2.0","id":2,"method":"tools/call",');
  });
  try {
    await served.close();
    await response;
  } finally {
    upload.destroy();
    await first.client.close();
  }
});

it("cancels a pending Evidence tool when the listener shuts down", async () => {
  const toolStart = phaseGate();
  const toolAbort = phaseArrival<unknown>();
  const { served } = await start({
    createMcpServer: () => {
      const server = createEchoServer(() => undefined);
      server.registerTool(
        "held",
        { inputSchema: z.strictObject({}) },
        async (_args, context) => {
          context.mcpReq.signal.addEventListener(
            "abort",
            () => toolAbort.arrive(context.mcpReq.signal.reason),
            { once: true },
          );
          await toolStart.hold();
          return { content: [] };
        },
      );
      return server;
    },
  });
  const first = await connect(served.url);
  const request = first.client
    .callTool({ name: "held", arguments: {} })
    .catch(() => undefined);
  let closing: Promise<void> | undefined;
  try {
    await toolStart.reached(request, "tool start");
    closing = served.prepareShutdown();
    await toolAbort.reached(closing, "tool abort");
  } finally {
    toolStart.release();
    await closeVerifierResources([
      () => first.client.close(),
      async () => {
        await Promise.allSettled([request, closing]);
      },
    ]);
  }
});
