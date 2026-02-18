# Open Router

https://openrouter.ai/

## Run

```
export OPENROUTER_API_KEY=your_api_key
./run.sh
```

## Result

Model: llama-3-3-70-b

```
❯ ./run.sh

> openrouter-poc@1.0.0 start
> ts-node --esm src/index.ts

(node:756) [DEP0180] DeprecationWarning: fs.Stats constructor is deprecated.
(Use `node --trace-deprecation ...` to show where the warning was created)
Enter a city name: Atlanta

Fetching curiosities about Atlanta...

Here are two interesting curiosities about the city of Atlanta:

1. **The World of Coca-Cola**: Located in downtown Atlanta, the World of Coca-Cola is a museum that showcases the history of the iconic brand. The museum features interactive exhibits, artifacts, and even a tasting room where you can sample flavors from around the world. What's particularly interesting is that the museum is housed in a 20,000-square-foot space that was once a Coca-Cola warehouse, highlighting the city's history with the brand.

2. **The Atlanta BeltLine**: Atlanta's BeltLine is an abandoned railway corridor that has been transformed into a vibrant network of parks, green spaces, and cultural attractions. The BeltLine stretches 22 miles through the city and has become a popular destination for hiking, biking, and exploring. What's interesting is that the BeltLine has become a symbol of Atlanta's creative reuse of underutilized spaces, demonstrating the city's ability to transform neglected areas into thriving community hubs.
```