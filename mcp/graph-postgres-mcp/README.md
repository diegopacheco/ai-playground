# GraphQL Postgres MCP

This MCP adds GraphQL layer into Postgres database. It allows you to query your Postgres database using GraphQL queries. The MCP does not access postgres directly, instead puts a Graphql in front.

## Usage

Start the Docker Database.
```
./start.sh 
```
Install the MCP
```
./install.sh
```

## Result

UI <br/>
<img src="ui-graphql-mcp-tool.png" width="600"/>

MCP in Codex <br/>
<img src="mcp.png" width="600"/>

What are my tables in postgres? <br/>
<img src="tables.png" width="600"/>

What are my cities with shipment ? <br/>
<img src="shipments.png" width="600"/>