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

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock window.matchMedia for components that use media queries
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock window.ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock window.IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
  takeRecords: vi.fn(),
}));

// Mock import.meta.env for Vite environment variables
vi.stubEnv('VITE_API_VDB_URL', 'http://localhost:8000');

// Mock CSS imports from KUI components and all CSS files
vi.mock('@kui/foundations/dist/**/*.css', () => ({}));
vi.mock('@kui/react/dist/**/*.css', () => ({}));

// Mock all CSS file imports 
Object.defineProperty(window, '__CSS_MODULES__', {
  value: {},
  writable: true,
});

// Mock specific KUI component CSS that might be missed
vi.mock('@kui/foundations/dist/components/accordion.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/anchor.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/badge.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/button.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/card.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/checkbox.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/divider.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/dropdown.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/form-field.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/grid.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/icon.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/menu.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/modal.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/panel.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/popover.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/radio.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/switch.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/tag.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/text.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/text-input.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/textarea.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/theme-provider.min.css', () => ({}));
vi.mock('@kui/foundations/dist/components/vertical-nav.min.css', () => ({})); 