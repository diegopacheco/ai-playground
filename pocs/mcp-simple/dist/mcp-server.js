"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const index_js_1 = require("@modelcontextprotocol/sdk/server/index.js");
const stdio_js_1 = require("@modelcontextprotocol/sdk/server/stdio.js");
const types_js_1 = require("@modelcontextprotocol/sdk/types.js");
const server = new index_js_1.Server({
    name: 'weather-mcp-server',
    version: '1.0.0',
}, {
    capabilities: {
        tools: {},
    },
});
const tools = [
    {
        name: 'get_weather',
        description: 'Get current weather for a location',
        inputSchema: {
            type: 'object',
            properties: {
                location: {
                    type: 'string',
                    description: 'City name or coordinates',
                },
            },
            required: ['location'],
        },
    },
    {
        name: 'calculate',
        description: 'Perform basic mathematical calculations',
        inputSchema: {
            type: 'object',
            properties: {
                expression: {
                    type: 'string',
                    description: 'Mathematical expression to evaluate',
                },
            },
            required: ['expression'],
        },
    },
];
server.setRequestHandler(types_js_1.ListToolsRequestSchema, async () => {
    return {
        tools,
    };
});
server.setRequestHandler(types_js_1.CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    switch (name) {
        case 'get_weather':
            const location = args?.location;
            if (!location) {
                return {
                    content: [
                        {
                            type: 'text',
                            text: 'Error: Location is required',
                        },
                    ],
                    isError: true,
                };
            }
            return {
                content: [
                    {
                        type: 'text',
                        text: `Weather in ${location}: 72°F, partly cloudy`,
                    },
                ],
            };
        case 'calculate':
            const expression = args?.expression;
            if (!expression) {
                return {
                    content: [
                        {
                            type: 'text',
                            text: 'Error: Expression is required',
                        },
                    ],
                    isError: true,
                };
            }
            try {
                const result = eval(expression);
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Result: ${result}`,
                        },
                    ],
                };
            }
            catch (error) {
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Error: ${error instanceof Error ? error.message : String(error)}`,
                        },
                    ],
                    isError: true,
                };
            }
        default:
            throw new Error(`Unknown tool: ${name}`);
    }
});
const transport = new stdio_js_1.StdioServerTransport();
server.connect(transport);
