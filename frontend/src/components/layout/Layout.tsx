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

import type { JSX } from "react";
import Header from "./Header";
import { ThemeProvider } from "@kui/react";
import { useTheme } from "../../hooks/useTheme";

/**
 * Main layout component that provides the application structure.
 * 
 * Renders the header and wraps child components in a consistent layout
 * with NVIDIA dark theme styling and responsive design.
 * 
 * @param props - Component props with children elements
 * @returns Layout wrapper with header and main content area
 */
const Layout = ({ children }: { children: JSX.Element | JSX.Element[] }) => {
  const { theme } = useTheme();
  
  return (
    <ThemeProvider theme={theme}>
      <Header/>
      <div style={{ 
        background: "var(--background-color-surface-base)",
        minHeight: "calc(100vh - 48px)",
      }}>
        {children}
      </div>
    </ThemeProvider>
  )
}

export default Layout;
