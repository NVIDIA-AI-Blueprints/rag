// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { render } from '@testing-library/react';
import type { RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import type { ReactElement, ReactNode } from 'react';

/**
 * Creates a new QueryClient instance for testing to avoid state pollution between tests.
 * 
 * @returns Configured QueryClient for testing
 */
const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
      },
      mutations: {
        retry: false,
      },
    },
  });

interface AllTheProvidersProps {
  children: ReactNode;
}

// Wrapper component that provides all necessary context providers  
/* eslint-disable-next-line react-refresh/only-export-components */
const AllTheProviders = ({ children }: AllTheProvidersProps) => {
  const queryClient = createTestQueryClient();

  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </BrowserRouter>
  );
};

/**
 * Custom render function that includes all necessary providers for testing.
 * 
 * Wraps components with React Query and React Router providers to simulate
 * the full application context during testing.
 * 
 * @param ui - React element to render
 * @param options - Additional render options
 * @returns Render result with full provider context
 */
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => render(ui, { wrapper: AllTheProviders, ...options });

/**
 * Re-export everything from @testing-library/react for convenience.
 */
/* eslint-disable react-refresh/only-export-components */
export * from '@testing-library/react';
/**
 * Export custom render function as the default render for tests.
 */
export { customRender as render };

// Export helper for creating test query clients
export { createTestQueryClient }; 