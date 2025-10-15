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
import type { HealthResponse } from "../types/api";

/**
 * Custom hook to fetch health status from the ingestor server.
 * 
 * @returns React Query object with health data
 */
export function useHealthStatus() {
  return useQuery<HealthResponse>({
    queryKey: ["health-status"],
    queryFn: async () => {
      const res = await fetch("/api/health?check_dependencies=true");
      if (!res.ok) throw new Error("Failed to fetch health status");
      return res.json();
    },
    staleTime: 30 * 1000, // Cache for 30 seconds
    retry: 1, // Only retry once to avoid delays
  });
}
