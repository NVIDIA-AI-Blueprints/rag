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

import type { MockedFunction } from 'vitest';

/**
 * Common mock types for testing utilities.
 */
export type MockedFetch = MockedFunction<typeof fetch>;

/**
 * Mock router parameter types for react-router-dom testing.
 */
export interface MockRouterParams {
  [key: string]: string | undefined;
}

/**
 * Mock location object for react-router testing.
 */
export interface MockLocation {
  pathname: string;
  search: string;
  hash: string;
  state: unknown;
  key: string;
}

/**
 * Helper type for test component props with optional data-testid.
 */
export type TestComponentProps<T = Record<string, unknown>> = T & {
  'data-testid'?: string;
};

/**
 * Common test data factory type for generating test data with overrides.
 */
export type TestDataFactory<T> = (overrides?: Partial<T>) => T; 