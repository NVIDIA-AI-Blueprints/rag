import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import Layout from '../Layout';

// Mock Header component to isolate Layout behavior
vi.mock('../Header', () => ({
  default: () => <div data-testid="header">Header</div>
}));

describe('Layout', () => {
  it('renders header component', () => {
    render(
      <Layout>
        <div>Content</div>
      </Layout>
    );
    
    expect(screen.getByTestId('header')).toBeInTheDocument();
  });

  it('renders children content', () => {
    render(
      <Layout>
        <div data-testid="child-content">Test Content</div>
      </Layout>
    );
    
    expect(screen.getByTestId('child-content')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('renders multiple children', () => {
    render(
      <Layout>
        <div data-testid="first-child">First</div>
        <div data-testid="second-child">Second</div>
      </Layout>
    );
    
    expect(screen.getByTestId('first-child')).toBeInTheDocument();
    expect(screen.getByTestId('second-child')).toBeInTheDocument();
  });
}); 