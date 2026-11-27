/**
 * Protocol revisions an embedded `createServer` accepts by default. Production
 * transports narrow this list to `MODERN_PROTOCOL_VERSIONS`, so the CLI never
 * serves the 2025-era handshake.
 */
export const DEFAULT_SUPPORTED_PROTOCOL_VERSIONS = [
  "2026-07-28",
  "2025-11-25",
  "2025-06-18",
  "2025-03-26",
  "2024-11-05",
  "2024-10-07",
] as const;

/** The only revisions a production stdio or HTTP endpoint serves. */
export const MODERN_PROTOCOL_VERSIONS = ["2026-07-28"] as const;
