import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8080/persons';

function App() {
  const [persons, setPersons] = useState([]);
  const [form, setForm] = useState({ name: '', email: '', age: '' });
  const [editingId, setEditingId] = useState(null);

  const fetchPersons = () => {
    fetch(API)
      .then(r => r.json())
      .then(setPersons)
      .catch(() => {});
  };

  useEffect(() => { fetchPersons(); }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    const body = JSON.stringify({ name: form.name, email: form.email, age: parseInt(form.age) });
    const headers = { 'Content-Type': 'application/json' };

    if (editingId) {
      fetch(`${API}/${editingId}`, { method: 'PUT', headers, body })
        .then(() => { resetForm(); fetchPersons(); });
    } else {
      fetch(API, { method: 'POST', headers, body })
        .then(() => { resetForm(); fetchPersons(); });
    }
  };

  const handleDelete = (id) => {
    fetch(`${API}/${id}`, { method: 'DELETE' })
      .then(() => fetchPersons());
  };

  const handleEdit = (person) => {
    setForm({ name: person.name, email: person.email, age: String(person.age) });
    setEditingId(person.id);
  };

  const resetForm = () => {
    setForm({ name: '', email: '', age: '' });
    setEditingId(null);
  };

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', fontFamily: 'Arial, sans-serif' }}>
      <h1>Person CRUD</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: 20, display: 'flex', gap: 10 }}>
        <input
          placeholder="Name"
          value={form.name}
          onChange={e => setForm({ ...form, name: e.target.value })}
          required
          style={{ padding: 8, flex: 1 }}
        />
        <input
          placeholder="Email"
          value={form.email}
          onChange={e => setForm({ ...form, email: e.target.value })}
          required
          style={{ padding: 8, flex: 1 }}
        />
        <input
          placeholder="Age"
          type="number"
          value={form.age}
          onChange={e => setForm({ ...form, age: e.target.value })}
          required
          style={{ padding: 8, width: 70 }}
        />
        <button type="submit" style={{ padding: '8px 16px' }}>
          {editingId ? 'Update' : 'Add'}
        </button>
        {editingId && (
          <button type="button" onClick={resetForm} style={{ padding: '8px 16px' }}>
            Cancel
          </button>
        )}
      </form>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid #333' }}>
            <th style={{ textAlign: 'left', padding: 8 }}>ID</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Name</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Email</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Age</th>
            <th style={{ textAlign: 'left', padding: 8 }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {persons.map(p => (
            <tr key={p.id} style={{ borderBottom: '1px solid #ccc' }}>
              <td style={{ padding: 8 }}>{p.id}</td>
              <td style={{ padding: 8 }}>{p.name}</td>
              <td style={{ padding: 8 }}>{p.email}</td>
              <td style={{ padding: 8 }}>{p.age}</td>
              <td style={{ padding: 8 }}>
                <button onClick={() => handleEdit(p)} style={{ marginRight: 5 }}>Edit</button>
                <button onClick={() => handleDelete(p.id)}>Delete</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
