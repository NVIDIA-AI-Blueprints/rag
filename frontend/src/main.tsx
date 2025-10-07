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

import "./index.css";

import ReactDOM from "react-dom/client";
import App from "./App";

import {
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";

/**
 * React Query client configuration for the application.
 */
const queryClient = new QueryClient();

/**
 * Application entry point that renders the React application.
 * 
 * Sets up the React root, React Query provider, and React Strict Mode
 * for the NVIDIA RAG frontend application.
 */
ReactDOM.createRoot(document.getElementById("root")!).render(
  // Temporary: Disabled StrictMode to fix double popover in development
  // <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  // </React.StrictMode>
);
