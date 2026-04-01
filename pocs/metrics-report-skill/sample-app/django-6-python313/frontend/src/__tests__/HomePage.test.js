import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import HomePage from '../pages/HomePage';

test('renders hero title', () => {
  render(<BrowserRouter><HomePage /></BrowserRouter>);
  expect(screen.getByText('Plan Your Retirement With Confidence')).toBeInTheDocument();
});

test('renders start simulation button', () => {
  render(<BrowserRouter><HomePage /></BrowserRouter>);
  expect(screen.getByText('Start Simulation')).toBeInTheDocument();
});

test('renders all four feature cards', () => {
  render(<BrowserRouter><HomePage /></BrowserRouter>);
  expect(screen.getByText('Compound Growth')).toBeInTheDocument();
  expect(screen.getByText('Inflation Adjusted')).toBeInTheDocument();
  expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
  expect(screen.getByText('Year-by-Year View')).toBeInTheDocument();
});
