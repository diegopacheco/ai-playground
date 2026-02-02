import { useState, useEffect } from 'react'

interface Product {
  id: number
  name: string
  price: number
  createdAt: string
  url: string
}

function App() {
  const [products, setProducts] = useState<Product[]>([])
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState({ name: '', price: '', createdAt: '', url: '' })

  useEffect(() => {
    fetchProducts()
  }, [])

  const fetchProducts = async () => {
    const response = await fetch('http://localhost:8000/api/products')
    const data = await response.json()
    setProducts(data)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await fetch('http://localhost:8000/api/products', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: formData.name,
        price: Number(formData.price),
        createdAt: formData.createdAt,
        url: formData.url
      })
    })
    setFormData({ name: '', price: '', createdAt: '', url: '' })
    setShowForm(false)
    fetchProducts()
  }

  const openProduct = (url: string) => {
    window.open(url, '_blank')
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '900px', margin: '0 auto' }}>
      <h1 style={{ color: '#333' }}>Product Manager</h1>

      <button
        onClick={() => setShowForm(!showForm)}
        style={{
          padding: '10px 20px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          marginBottom: '20px'
        }}
      >
        {showForm ? 'Cancel' : 'Add New Product'}
      </button>

      {showForm && (
        <form onSubmit={handleSubmit} style={{ marginBottom: '20px', padding: '20px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
          <div style={{ marginBottom: '10px' }}>
            <input
              type="text"
              placeholder="Product Name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              style={{ padding: '8px', width: '200px', marginRight: '10px' }}
            />
          </div>
          <div style={{ marginBottom: '10px' }}>
            <input
              type="number"
              placeholder="Price"
              value={formData.price}
              onChange={(e) => setFormData({ ...formData, price: e.target.value })}
              required
              style={{ padding: '8px', width: '200px', marginRight: '10px' }}
            />
          </div>
          <div style={{ marginBottom: '10px' }}>
            <input
              type="date"
              value={formData.createdAt}
              onChange={(e) => setFormData({ ...formData, createdAt: e.target.value })}
              required
              style={{ padding: '8px', width: '200px', marginRight: '10px' }}
            />
          </div>
          <div style={{ marginBottom: '10px' }}>
            <input
              type="url"
              placeholder="Product URL"
              value={formData.url}
              onChange={(e) => setFormData({ ...formData, url: e.target.value })}
              required
              style={{ padding: '8px', width: '300px', marginRight: '10px' }}
            />
          </div>
          <button
            type="submit"
            style={{
              padding: '10px 20px',
              backgroundColor: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            Save Product
          </button>
        </form>
      )}

      <table style={{ width: '100%', borderCollapse: 'collapse', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <thead>
          <tr style={{ backgroundColor: '#007bff', color: 'white' }}>
            <th style={{ padding: '12px', textAlign: 'left' }}>Name</th>
            <th style={{ padding: '12px', textAlign: 'left' }}>Price</th>
            <th style={{ padding: '12px', textAlign: 'left' }}>Created At</th>
            <th style={{ padding: '12px', textAlign: 'left' }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {products.map((product, index) => (
            <tr
              key={product.id}
              style={{
                backgroundColor: index % 2 === 0 ? '#fff' : '#f9f9f9',
                cursor: 'pointer'
              }}
              onClick={() => openProduct(product.url)}
            >
              <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>{product.name}</td>
              <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>${product.price}</td>
              <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>{product.createdAt}</td>
              <td style={{ padding: '12px', borderBottom: '1px solid #ddd' }}>
                <button
                  onClick={(e) => { e.stopPropagation(); openProduct(product.url); }}
                  style={{
                    padding: '5px 10px',
                    backgroundColor: '#17a2b8',
                    color: 'white',
                    border: 'none',
                    borderRadius: '3px',
                    cursor: 'pointer'
                  }}
                >
                  View
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default App
