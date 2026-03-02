# Dapr Agents

https://docs.dapr.io/developing-ai/dapr-agents/dapr-agents-introduction/

## How to run
```bash
OPENAI_API_KEY=sk-xxx ./run.sh
```

## Results

```
ℹ️  Checking if Dapr sidecar is listening on GRPC port 56082
ℹ️  Dapr sidecar is up and running.
ℹ️  Updating metadata for appPID: 72440
ℹ️  Updating metadata for app command: python agent.py
✅  You're up and running! Both Dapr and your app logs will appear here.


User: What is the weather in New York?
user:
What is the weather in New York?

--------------------------------------------------------------------------------

AssistantAgent(assistant):
Function name: GetWeather (Call Id: call_sTPielv1N9aVnB3UmtgAWguH)
Arguments: {"location":"New York"}

--------------------------------------------------------------------------------

GetWeather(tool) (Id: call_sTPielv1N9aVnB3UmtgAWguH):
New York: 62F, sunny.

--------------------------------------------------------------------------------

AssistantAgent(assistant):
The weather in New York is currently 62°F and sunny. Enjoy your day!

--------------------------------------------------------------------------------

AssistantAgent(assistant):
User asked for the weather in New York. Tool provided current weather as 62°F and sunny. Assistant relayed this information to the user.

--------------------------------------------------------------------------------

Agent: content='The weather in New York is currently 62°F and sunny. Enjoy your day!' role='assistant'

User: What time is it in PST?
user:
What time is it in PST?

--------------------------------------------------------------------------------

AssistantAgent(assistant):
Function name: GetTime (Call Id: call_ZNLyLZwFzyFQQvfI3N8sp57N)
Arguments: {"timezone":"PST"}

--------------------------------------------------------------------------------

GetTime(tool) (Id: call_ZNLyLZwFzyFQQvfI3N8sp57N):
Current time in PST: 2026-03-01 20:43:36

--------------------------------------------------------------------------------

AssistantAgent(assistant):
The current time in PST is 8:43 PM.

--------------------------------------------------------------------------------

AssistantAgent(assistant):
User inquired about the current time in PST. A tool was used to retrieve the time, which was found to be 8:43 PM on March 1, 2026. The outcome confirmed the requested information.

--------------------------------------------------------------------------------

Agent: content='The current time in PST is 8:43 PM.' role='assistant'

User: How about the weather in Tokyo and the time in JST?
user:
How about the weather in Tokyo and the time in JST?

--------------------------------------------------------------------------------

AssistantAgent(assistant):
Function name: GetWeather (Call Id: call_zmmxClGz4cZToVwJ8SpbaA3H)
Arguments: {"location": "Tokyo"}

--------------------------------------------------------------------------------

AssistantAgent(assistant):
Function name: GetTime (Call Id: call_1F339blTGJwAalKcxJFo6Hym)
Arguments: {"timezone": "Asia/Tokyo"}

--------------------------------------------------------------------------------

GetWeather(tool) (Id: call_zmmxClGz4cZToVwJ8SpbaA3H):
Tokyo: 65F, snowy.

--------------------------------------------------------------------------------

GetTime(tool) (Id: call_1F339blTGJwAalKcxJFo6Hym):
Current time in Asia/Tokyo: 2026-03-02 04:43:41

--------------------------------------------------------------------------------

AssistantAgent(assistant):
The weather in Tokyo is 65°F and snowy. The current time in Japan Standard Time (JST) is 04:43 AM on March 2, 2026.

--------------------------------------------------------------------------------

AssistantAgent(assistant):
User inquired about the weather and current time in Tokyo. The response provided included that the weather was 65°F and snowy, and the time was 04:43 AM on March 2, 2026. Two tools were used: one to retrieve the weather information and another for the current time in JST.

--------------------------------------------------------------------------------

Agent: content='The weather in Tokyo is 65°F and snowy. The current time in Japan Standard Time (JST) is 04:43 AM on March 2, 2026.' role='assistant'
✅  Exited App successfully
ℹ️
terminated signal received: shutting down
✅  Exited Dapr successfully
✅  Exited App successfully

Agent finished. Dashboard still running at http://localhost:9999
Press Ctrl+C to stop the dashboard.
```
