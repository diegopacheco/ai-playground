import { apiService } from "./api";
import { Product, CreateProductRequest, UpdateProductRequest } from "@/types";

export class ProductService {
  static async getAll(): Promise<Product[]> {
    return apiService.get<Product[]>("/products");
  }

  static async getById(id: number): Promise<Product> {
    return apiService.get<Product>(`/products/${id}`);
  }

  static async create(data: CreateProductRequest): Promise<Product> {
    return apiService.post<Product>("/products", data);
  }

  static async update(
    id: number,
    data: UpdateProductRequest
  ): Promise<Product> {
    return apiService.put<Product>(`/products/${id}`, data);
  }

  static async delete(id: number): Promise<{ message: string }> {
    return apiService.delete<{ message: string }>(`/products/${id}`);
  }
}

export default ProductService;
