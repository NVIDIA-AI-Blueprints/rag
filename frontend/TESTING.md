# Testing Guide

This project uses **Vitest** with **React Testing Library** for comprehensive testing of React components, hooks, and utilities.

## 🚀 Quick Start

```bash
# Run all tests
pnpm test

# Run tests in watch mode
pnpm test:watch

# Run tests once (CI mode)
pnpm test:run

# Run tests with coverage
pnpm test:coverage

# Open Vitest UI
pnpm test:ui
```

## 📁 Project Structure

```
src/
├── test/              # Testing utilities and setup
│   ├── setup.ts       # Global test configuration
│   ├── utils.tsx      # Custom render functions & providers
│   ├── types.ts       # Testing-specific types
│   └── example.test.tsx # Example test showcasing patterns
├── components/
│   └── **/__tests__/  # Component tests
├── hooks/
│   └── **/__tests__/  # Hook tests
└── pages/
    └── **/__tests__/  # Page tests
```

## 🔧 Configuration

### Vitest Config (`vitest.config.ts`)
- Extends the main Vite configuration
- Uses jsdom environment for DOM testing
- Includes path aliases for clean imports
- Configured for comprehensive coverage reporting

### Test Setup (`src/test/setup.ts`)
- Global test utilities and mocks
- DOM environment setup (matchMedia, ResizeObserver, etc.)
- Environment variable mocking
- Automatic cleanup after each test

## 🧪 Testing Patterns

### Basic Component Testing

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/utils';
import { MyComponent } from './MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent title="Hello World" />);
    
    expect(screen.getByText('Hello World')).toBeInTheDocument();
  });
});
```

### Testing with User Interactions

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@/test/utils';
import userEvent from '@testing-library/user-event';
import { ButtonComponent } from './ButtonComponent';

describe('ButtonComponent', () => {
  it('handles click events', async () => {
    const user = userEvent.setup();
    const handleClick = vi.fn();
    
    render(<ButtonComponent onClick={handleClick} />);
    
    await user.click(screen.getByRole('button'));
    
    expect(handleClick).toHaveBeenCalledOnce();
  });
});
```

### Mocking External Dependencies

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/utils';

// Mock external hook
vi.mock('@/hooks/useCustomHook', () => ({
  useCustomHook: () => ({
    data: { message: 'mocked data' },
    loading: false,
    error: null
  })
}));

describe('ComponentWithHook', () => {
  it('displays mocked data', () => {
    render(<ComponentWithHook />);
    
    expect(screen.getByText('mocked data')).toBeInTheDocument();
  });
});
```

### Testing React Router Components

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/test/utils';

// Mock react-router-dom
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => ({ id: '123' }),
    useNavigate: () => vi.fn(),
  };
});

describe('RoutedComponent', () => {
  it('uses route parameters', () => {
    render(<RoutedComponent />);
    
    expect(screen.getByText('ID: 123')).toBeInTheDocument();
  });
});
```

### Testing API Calls

```tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/test/utils';

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('DataFetcher', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('displays fetched data', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ name: 'John Doe' })
    });

    render(<DataFetcher />);

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });

    expect(mockFetch).toHaveBeenCalledWith('/api/users');
  });
});
```

### Testing Custom Hooks

```tsx
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useCounter } from './useCounter';

describe('useCounter', () => {
  it('increments counter', () => {
    const { result } = renderHook(() => useCounter(0));

    expect(result.current.count).toBe(0);

    act(() => {
      result.current.increment();
    });

    expect(result.current.count).toBe(1);
  });
});
```

## 🛠 Testing Utilities

### Custom Render Function

Our custom `render` function from `@/test/utils` automatically wraps components with necessary providers:

- React Router (`BrowserRouter`)
- React Query (`QueryClientProvider`)

```tsx
// Automatically includes all providers
render(<MyComponent />);

// For components that don't need providers, use RTL directly
import { render as rtlRender } from '@testing-library/react';
rtlRender(<SimpleComponent />);
```

### Environment Setup

The test environment includes mocks for:

- `window.matchMedia` - For responsive design testing
- `window.ResizeObserver` - For component resize observations
- `window.IntersectionObserver` - For intersection-based functionality
- `import.meta.env.VITE_API_VDB_URL` - API endpoint configuration

## 📋 Best Practices

### 1. Test Structure
```tsx
describe('ComponentName', () => {
  describe('Rendering', () => {
    // Basic rendering tests
  });

  describe('User Interactions', () => {
    // Click, input, keyboard tests
  });

  describe('Props', () => {
    // Props validation and behavior
  });

  describe('Edge Cases', () => {
    // Error states, empty data, etc.
  });
});
```

### 2. Query Priorities
1. **Role-based queries**: `getByRole('button', { name: 'Submit' })`
2. **Label queries**: `getByLabelText('Email address')`
3. **Text content**: `getByText('Welcome')`
4. **Test IDs**: `getByTestId('user-card')` (last resort)

### 3. Async Testing
```tsx
// Wait for elements to appear
await waitFor(() => {
  expect(screen.getByText('Loading complete')).toBeInTheDocument();
});

// Wait for elements to disappear
await waitForElementToBeRemoved(screen.getByText('Loading...'));
```

### 4. Mock Management
```tsx
// Reset mocks between tests
beforeEach(() => {
  vi.clearAllMocks();
});

// Restore original implementations after tests
afterAll(() => {
  vi.restoreAllMocks();
});
```

## 🎯 Coverage Goals

- **Statements**: > 80%
- **Branches**: > 75%
- **Functions**: > 80%
- **Lines**: > 80%

## 🚦 CI/CD Integration

Tests are configured to run in CI with:
```bash
pnpm test:run --coverage
```

Coverage reports are generated in multiple formats:
- Terminal output for immediate feedback
- HTML report in `coverage/` directory
- JSON/Clover formats for CI integration

## 📚 Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Testing Library Queries](https://testing-library.com/docs/queries/about/)
- [Common Testing Patterns](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library) 