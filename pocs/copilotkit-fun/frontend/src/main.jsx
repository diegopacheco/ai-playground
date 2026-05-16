import React from 'react';
import { createRoot } from 'react-dom/client';
import { CopilotKit } from '@copilotkit/react-core';
import '@copilotkit/react-ui/styles.css';
import App from './App.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <CopilotKit runtimeUrl="/api/copilotkit" useSingleEndpoint={false}>
    <App />
  </CopilotKit>
);
