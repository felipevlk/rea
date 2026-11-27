import {
  Client,
  StreamableHTTPClientTransport,
  type StreamableHTTPClientTransportOptions,
} from "@modelcontextprotocol/client";
import { describe, expect, onTestFinished } from "vitest";
import { once } from "node:events";
import { resolve as resolvePath } from "node:path";

import { advertisedMcpToolSchemas } from "../../fixtures/advertisedMcpTool.js";
import { expectForeignHostRefused } from "../../support/http/rawRequest.js";
import {
  startListeningProcess,
  startMcpHttpProcess,
} from "../../support/mcp/streamableHttpProcess.js";
import { processTest } from "../../support/process/processFixture.js";
import { assertAdvertisedMcpContracts } from "./mcpContractHarness.js";
import { createMcpSchemaValidation } from "../../fixtures/mcpSchemaValidation.js";

/** Open one 2026-07-28 client; the modern era has no MCP session to negotiate. */
const openClient = async (
  url: URL,
  name: string,
  options?: StreamableHTTPClientTransportOptions,
) => {
  const transport = new StreamableHTTPClientTransport(url, options);
  const client = new Client(
    { name, version: "1" },
    {
      jsonSchemaValidator: createMcpSchemaValidation(),
      versionNegotiation: { mode: { pin: "2026-07-28" } },
    },
  );
  onTestFinished(() => client.close());
  await client.connect(transport);
  expect(transport.sessionId).toBeUndefined();
  const { toolDefinition } = await advertisedMcpToolSchemas(
    client,
    "binary_session",
  );
  assertAdvertisedMcpContracts(["binary_session"], [toolDefinition]);
  return { transport, client, toolDefinition };
};

/** The shared target answers with its identity only while its providers are still open. */
const expectIdentity = async ({
  client,
  toolDefinition,
}: Awaited<ReturnType<typeof openClient>>): Promise<void> => {
  const identity = await client.callTool(
    { name: "binary_session", arguments: {} },
    { toolDefinition },
  );
  expect(identity.isError).toBeFalsy();
  expect(identity.structuredContent).toMatchObject({
    result: {
      server_identity: { package: { version: expect.any(String) } },
    },
  });
};

describe("rea mcp --http", () => {
  processTest(
    "serves server/discover, tools/list and a tool call without an MCP session",
    async ({ processes }) => {
      const started = await startListeningProcess(
        processes,
        ["mcp", "--http", "--port", "0"],
        // Sessions validate advertised output schemas, which default to off.
        {},
      );
      expect(started.url.hostname).toBe("127.0.0.1");
      expect(started.url.pathname).toBe("/mcp");
      let stdout = "";
      started.child.stdout.on("data", (chunk: Buffer) => {
        stdout += chunk.toString("utf8");
      });
      const first = await openClient(started.url, "streamable-http-boundary");
      const names = (await first.client.listTools()).tools.map(
        ({ name }) => name,
      );
      // The stateless endpoint advertises the complete catalog on every request.
      expect(names).toContain("binary_session");
      expect(names).toContain("inspect_analysis_view");
      await expectIdentity(first);
      await first.client.close();
      // A later request gets a fresh instance but the same shared target session.
      const second = await openClient(started.url, "next-client");
      await expectIdentity(second);
      await second.client.close();
      expect(stdout).toBe("");
      expect(started.stderr()).toContain('"transport":"streamable-http"');
    },
  );

  processTest(
    "refuses the per-connection hot/cold catalog on the stateless HTTP transport",
    async ({ processes }) => {
      const child = processes.spawn(
        process.execPath,
        [resolvePath("scripts/rea.mjs"), "mcp", "--http", "--port", "0"],
        {
          env: {
            PATH: process.env.PATH ?? "",
            HOPPER_LAUNCHER_PATH: "/rea-unconfigured-deep-provider/hopper",
            REA_MCP_TOOL_CATALOG_PROFILE: "hot-cold",
          },
        },
      );
      child.stdin.end();
      let stderr = "";
      child.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString("utf8");
      });
      const [code] = await once(child, "exit");
      expect(code).toBe(1);
      expect(stderr).toContain("REA_MCP_TOOL_CATALOG_PROFILE=hot-cold");
      expect(stderr).toContain("stateless");
    },
  );

  processTest(
    "refuses a non-loopback Host header on the loopback bind",
    async ({ processes }) => {
      const { url } = await startMcpHttpProcess(processes);
      const refused = await expectForeignHostRefused(url);
      expect(refused.body).not.toContain("127.0.0.1");
    },
  );
});
