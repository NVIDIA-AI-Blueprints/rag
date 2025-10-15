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

import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Chat from "./pages/Chat";
import NewCollection from "./pages/NewCollection";
import Layout from "./components/layout/Layout";
import SettingsPage from "./pages/SettingsPage";
import { useAppHealthStatus } from "./store/useSettingsStore";
import { useHealthMonitoring } from "./hooks/useHealthMonitoring";

/**
 * React Query client configuration with default options.
 */
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

/**
 * App content component that initializes settings and monitors health.
 */
function AppContent() {
  // Get application health status data (Note: may be redundant since useHealthMonitoring also fetches health data)
  useAppHealthStatus();
  
  // Initialize model settings from health endpoint data
  // useHealthInitialization(); // Disabled: Models should start empty like other settings
  
  // Monitor service health and create notifications for issues
  useHealthMonitoring();
  
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Chat />} />
        <Route path="/collections/new" element={<NewCollection />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </Layout>
  );
}

/**
 * Main application component that sets up routing and provides React Query context.
 * 
 * @returns The main App component with routing and layout
 */
export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppContent />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
