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

import { useQuery } from "@tanstack/react-query";
import type { ConfigurationResponse } from "../types/api";

/**
 * Custom hook to fetch server configuration defaults from the RAG server.
 * 
 * Returns the actual default values configured on the server, allowing the UI
 * to display accurate defaults instead of hardcoded placeholder values.
 * 
 * @returns React Query object with configuration data
 * 
 * @example
 * ```tsx
 * const { data: config, isLoading, error } = useServerConfiguration();
 * if (config) {
 *   console.log('Default temperature:', config.rag_configuration.temperature);
 * }
 * ```
 */
export function useServerConfiguration() {
  return useQuery<ConfigurationResponse>({
    queryKey: ["server-configuration"],
    queryFn: async () => {
      const res = await fetch("/api/configuration");
      if (!res.ok) throw new Error("Failed to fetch server configuration");
      return res.json();
    },
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes (config rarely changes)
    retry: 2, // Retry twice on failure
  });
}

