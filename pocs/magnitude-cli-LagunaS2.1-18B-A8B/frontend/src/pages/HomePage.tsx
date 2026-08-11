import {
  Container,
  PageWrapper,
  Section,
  SectionTitle,
  FlexBetween,
  FlexGap,
} from "@/components/layout/Container/Container";
import { Header } from "@/components/layout/Header/Header";
import { Button } from "@/components/common/Button/Button";
import { ProductTable } from "@/components/table/ProductTable";
import { ProductForm } from "@/components/forms/ProductForm";
import { Modal } from "@/components/common/Modal/Modal";
import { useProducts } from "@/hooks";
import { Product } from "@/types";
import { useEffect, useState } from "react";

type ModalMode = "create" | "edit" | null;

export function HomePage() {
  const {
    products,
    loading,
    error,
    fetchProducts,
    createProduct,
    updateProduct,
    deleteProduct,
  } = useProducts();

  const [modalMode, setModalMode] = useState<ModalMode>(null);
  const [editingProduct, setEditingProduct] = useState<Product | null>(null);
  const [formLoading, setFormLoading] = useState(false);

  useEffect(() => {
    fetchProducts();
  }, [fetchProducts]);

  const handleCreate = () => {
    setEditingProduct(null);
    setModalMode("create");
  };

  const handleEdit = (product: Product) => {
    setEditingProduct(product);
    setModalMode("edit");
  };

  const handleDelete = async (product: Product) => {
    if (!confirm(`Delete "${product.name}"?`)) return;
    try {
      await deleteProduct(product.id);
    } catch {
      // Error is handled in the hook
    }
  };

  const handleSubmit = async (data: any) => {
    setFormLoading(true);
    try {
      if (modalMode === "edit" && editingProduct) {
        await updateProduct(editingProduct.id, data);
      } else {
        await createProduct(data);
      }
      setModalMode(null);
      setEditingProduct(null);
    } catch {
      // Error is handled in the hook
    } finally {
      setFormLoading(false);
    }
  };

  const getModalTitle = () => {
    if (modalMode === "edit") return "Edit Product";
    return "Add New Product";
  };

  const getFormInitialData = () => {
    if (modalMode === "edit" && editingProduct) {
      return {
        name: editingProduct.name,
        description: editingProduct.description,
        price: editingProduct.price,
        category: editingProduct.category,
        in_stock: editingProduct.in_stock,
      };
    }
    return undefined;
  };

  return (
    <PageWrapper>
      <Header
        title="Modular CRUD App"
        subtitle="Product Catalog — React 19 • TypeScript • TanStack Table • styled-components"
        actions={
          <Button variant="primary" onClick={handleCreate}>
            + Add Product
          </Button>
        }
      />

      <Container>
        <Section>
          <FlexBetween style={{ marginBottom: "1rem" }}>
            <SectionTitle>Products ({products.length})</SectionTitle>
            {error && (
              <span style={{ color: "#ef4444", fontSize: "0.875rem" }}>
                ⚠️ {error}
              </span>
            )}
          </FlexBetween>

          {loading && products.length === 0 ? (
            <div style={{ textAlign: "center", padding: "3rem" }}>
              Loading products...
            </div>
          ) : (
            <ProductTable
              products={products}
              onEdit={handleEdit}
              onDelete={handleDelete}
            />
          )}
        </Section>
      </Container>

      <Modal
        isOpen={modalMode !== null}
        onClose={() => {
          setModalMode(null);
          setEditingProduct(null);
        }}
        title={getModalTitle()}
        footer={
          <FlexGap>
            <Button
              type="button"
              variant="secondary"
              onClick={() => {
                setModalMode(null);
                setEditingProduct(null);
              }}
              disabled={formLoading}
            >
              Cancel
            </Button>
            <Button type="submit" variant="primary" disabled={formLoading}>
              {formLoading ? "Saving..." : "Save"}
            </Button>
          </FlexGap>
        }
      >
        <ProductForm
          key={modalMode === "edit" ? editingProduct?.id : "create"}
          initialData={getFormInitialData()}
          onSubmit={handleSubmit}
          onCancel={() => {
            setModalMode(null);
            setEditingProduct(null);
          }}
          loading={formLoading}
          submitLabel={modalMode === "edit" ? "Update Product" : "Create Product"}
        />
      </Modal>
    </PageWrapper>
  );
}
