export interface Product {
  id: number;
  name: string;
  description: string | null;
  price: number;
  category: string;
  in_stock: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface CreateProductRequest {
  name: string;
  description?: string | null;
  price: number;
  category: string;
  in_stock?: boolean;
}

export interface UpdateProductRequest extends CreateProductRequest {}

export interface ApiResponse<T> {
  data: T;
}

export interface ApiError {
  error: string;
  details?: string[];
}
