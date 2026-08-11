import { useState, useEffect } from "react";
import {
  Form,
  FormGroup,
  FormSection,
  CheckboxRow,
} from "./form.styles";
import {
  InputWrapper,
  Label,
  Input,
  Textarea,
  Select,
  Checkbox,
  ErrorMessage,
} from "@/components/common/Input/Input";
import { Button } from "@/components/common/Button/Button";
import { Product, CreateProductRequest, UpdateProductRequest } from "@/types";

export interface ProductFormProps {
  initialData?: Partial<CreateProductRequest>;
  onSubmit: (data: CreateProductRequest | UpdateProductRequest) => Promise<void>;
  onCancel: () => void;
  loading?: boolean;
  submitLabel?: string;
}

const CATEGORIES = [
  "Electronics",
  "Clothing",
  "Food & Beverage",
  "Books",
  "Home & Garden",
  "Toys & Games",
  "Sports & Outdoors",
  "Beauty & Health",
  "Office Supplies",
  "Other",
];

export function ProductForm({
  initialData,
  onSubmit,
  onCancel,
  loading = false,
  submitLabel = "Save Product",
}: ProductFormProps) {
  const [name, setName] = useState(initialData?.name ?? "");
  const [description, setDescription] = useState(
    initialData?.description ?? ""
  );
  const [price, setPrice] = useState(
    initialData?.price !== undefined ? String(initialData.price) : ""
  );
  const [category, setCategory] = useState(initialData?.category ?? "Other");
  const [inStock, setInStock] = useState(initialData?.in_stock ?? true);
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (initialData) {
      setName(initialData.name ?? "");
      setDescription(initialData.description ?? "");
      setPrice(
        initialData.price !== undefined
          ? String(initialData.price)
          : ""
      );
      setCategory(initialData.category ?? "Other");
      setInStock(initialData.in_stock ?? true);
    }
  }, [initialData]);

  const validate = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!name.trim()) {
      newErrors.name = "Name is required";
    }

    if (!price.trim()) {
      newErrors.price = "Price is required";
    } else {
      const numPrice = parseFloat(price);
      if (isNaN(numPrice)) {
        newErrors.price = "Price must be a valid number";
      } else if (numPrice < 0) {
        newErrors.price = "Price must be non-negative";
      }
    }

    if (!category) {
      newErrors.category = "Category is required";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) return;

    const data: CreateProductRequest = {
      name: name.trim(),
      description: description.trim() || null,
      price: parseFloat(price),
      category,
      in_stock: inStock,
    };

    await onSubmit(data);
  };

  return (
    <Form onSubmit={handleSubmit}>
      <FormGroup data-full-width>
        <Label htmlFor="name">Name *</Label>
        <Input
          id="name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Product name"
          disabled={loading}
        />
        {errors.name && <ErrorMessage>{errors.name}</ErrorMessage>}
      </FormGroup>

      <FormGroup data-full-width>
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Product description (optional)"
          disabled={loading}
        />
      </FormGroup>

      <FormGroup>
        <Label htmlFor="price">Price *</Label>
        <Input
          id="price"
          type="number"
          step="0.01"
          min="0"
          value={price}
          onChange={(e) => setPrice(e.target.value)}
          placeholder="0.00"
          disabled={loading}
        />
        {errors.price && <ErrorMessage>{errors.price}</ErrorMessage>}
      </FormGroup>

      <FormGroup>
        <Label htmlFor="category">Category *</Label>
        <Select
          id="category"
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          disabled={loading}
        >
          {CATEGORIES.map((cat) => (
            <option key={cat} value={cat}>
              {cat}
            </option>
          ))}
        </Select>
        {errors.category && <ErrorMessage>{errors.category}</ErrorMessage>}
      </FormGroup>

      <FormSection>
        <CheckboxRow>
          <Checkbox
            id="in_stock"
            checked={inStock}
            onChange={(e) => setInStock(e.target.checked)}
            disabled={loading}
          />
          <Label htmlFor="in_stock" style={{ cursor: "pointer" }}>
            In Stock
          </Label>
        </CheckboxRow>
      </FormSection>

      <FormSection
        style={{
          display: "flex",
          justifyContent: "flex-end",
          gap: "0.75rem",
        }}
      >
        <Button
          type="button"
          variant="secondary"
          onClick={onCancel}
          disabled={loading}
        >
          Cancel
        </Button>
        <Button type="submit" variant="primary" disabled={loading}>
          {loading ? "Saving..." : submitLabel}
        </Button>
      </FormSection>
    </Form>
  );
}
