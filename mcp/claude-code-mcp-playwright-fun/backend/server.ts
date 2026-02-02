const products: Product[] = [
  { id: 1, name: "iPhone 15", price: 999, createdAt: "2023-09-22", url: "https://www.apple.com/iphone" },
  { id: 2, name: "MacBook Pro", price: 1999, createdAt: "2023-10-30", url: "https://www.apple.com/macbook-pro" },
  { id: 3, name: "Apple Watch", price: 399, createdAt: "2023-09-12", url: "https://www.apple.com/watch" },
];

interface Product {
  id: number;
  name: string;
  price: number;
  createdAt: string;
  url: string;
}

let nextId = 4;

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  if (url.pathname === "/" && req.method === "GET") {
    return new Response(JSON.stringify({ message: "Product API", endpoints: ["/api/products"] }), {
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  if (url.pathname === "/api/products" && req.method === "GET") {
    return new Response(JSON.stringify(products), {
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  if (url.pathname === "/api/products" && req.method === "POST") {
    const body = await req.json();
    const newProduct: Product = {
      id: nextId++,
      name: body.name,
      price: body.price,
      createdAt: body.createdAt,
      url: body.url,
    };
    products.push(newProduct);
    return new Response(JSON.stringify(newProduct), {
      status: 201,
      headers: { "Content-Type": "application/json", ...corsHeaders },
    });
  }

  return new Response("Not Found", { status: 404, headers: corsHeaders });
}

console.log("Server running on http://localhost:8000");
Deno.serve({ port: 8000 }, handler);
