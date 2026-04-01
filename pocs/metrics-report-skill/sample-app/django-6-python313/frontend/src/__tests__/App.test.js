import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';

test('renders navbar brand', () => {
  render(<App />);
  expect(screen.getByText('RetireSmart')).toBeInTheDocument();
});

test('renders Home nav link', () => {
  render(<App />);
  expect(screen.getByText('Home')).toBeInTheDocument();
});

test('renders Simulate nav link', () => {
  render(<App />);
  expect(screen.getByText('Simulate')).toBeInTheDocument();
});

test('renders Results nav link', () => {
  render(<App />);
  expect(screen.getByText('Results')).toBeInTheDocument();
});

test('renders About nav link', () => {
  render(<App />);
  expect(screen.getByText('About')).toBeInTheDocument();
});
