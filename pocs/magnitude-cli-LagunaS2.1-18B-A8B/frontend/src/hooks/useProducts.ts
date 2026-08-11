import { useCallback, useState } from "react";
import { Product, CreateProductRequest, UpdateProductRequest } from "@/types";
import { ProductService } from "@/services";

export interface UseProductsReturn {
  products: Product[];
  loading: boolean;
  error: string | null;
  fetchProducts: () => Promise<void>;
  createProduct: (data: CreateProductRequest) => Promise<Product>;
  updateProduct: (id: number, data: UpdateProductRequest) => Promise<Product>;
  deleteProduct: (id: number) => Promise<void>;
}

export function useProducts(): UseProductsReturn {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchProducts = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await ProductService.getAll();
      setProducts(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to fetch products"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const createProduct = useCallback(
    async (data: CreateProductRequest): Promise<Product> => {
      setError(null);
      try {
        const newProduct = await ProductService.create(data);
        setProducts((prev) => [newProduct, ...prev]);
        return newProduct;
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to create product";
        setError(msg);
        throw err;
      }
    },
    []
  );

  const updateProduct = useCallback(
    async (id: number, data: UpdateProductRequest): Promise<Product> => {
      setError(null);
      try {
        const updated = await ProductService.update(id, data);
        setProducts((prev) =>
          prev.map((p) => (p.id === id ? updated : p))
        );
        return updated;
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to update product";
        setError(msg);
        throw err;
      }
    },
    []
  );

  const deleteProduct = useCallback(
    async (id: number): Promise<void> => {
      setError(null);
      try {
        await ProductService.delete(id);
        setProducts((prev) => prev.filter((p) => p.id !== id));
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to delete product";
        setError(msg);
        throw err;
      }
    },
    []
  );

  return {
    products,
    loading,
    error,
    fetchProducts,
    createProduct,
    updateProduct,
    deleteProduct,
  };
}
