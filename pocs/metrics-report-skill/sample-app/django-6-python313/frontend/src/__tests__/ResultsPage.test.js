import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import ResultsPage from '../pages/ResultsPage';

test('shows no results message when results is null', () => {
  render(<BrowserRouter><ResultsPage results={null} inputData={null} /></BrowserRouter>);
  expect(screen.getByText('No Results Yet')).toBeInTheDocument();
});

test('shows link to simulation when no results', () => {
  render(<BrowserRouter><ResultsPage results={null} inputData={null} /></BrowserRouter>);
  expect(screen.getByText('simulation')).toBeInTheDocument();
});
