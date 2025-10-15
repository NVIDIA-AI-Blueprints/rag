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

import { useSidebarStore } from "../../store/useSidebarStore";
import Citations from "../chat/Citations";
import { 
  SidePanel, 
  Flex, 
  Text, 
  Block, 
  Stack,
  Button
} from "@kui/react";

const CitationsIcon = () => (
  <svg 
    style={{ width: '20px', height: '20px' }} 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const CloseIcon = () => (
  <svg 
    style={{ width: '20px', height: '20px' }} 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

export default function SidebarDrawer() {
  const { view, citations, closeSidebar } = useSidebarStore();

  return (
    <SidePanel
      open={!!view}
      onOpenChange={(open) => {
        if (!open) {
          closeSidebar();
        }
      }}
      side="right"
      modal
      closeOnClickOutside
      hideCloseButton
      style={{
        "--side-panel-width": "75vw",
        backgroundColor: "var(--background-color-interaction-inverse)"
      }}
      slotHeading={
        <Flex align="center" justify="between" gap="3">
          <Flex align="center" gap="3">
            <Block
              style={{  
                padding: '8px',
                borderRadius: '8px',
                backgroundColor: 'var(--background-color-accent-green-subtle)'
              }}
            >
              <CitationsIcon />
            </Block>
            <Stack>
              <Text kind="body/bold/lg" style={{ color: 'var(--text-color-inverse)' }}>
                {view === "citations" && "Source Citations"}
              </Text>
              <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                {view === "citations" && `${citations?.length || 0} sources found`}
              </Text>
            </Stack>
          </Flex>
          <Button
            kind="tertiary"
            size="small"
            onClick={closeSidebar}
            aria-label="Close sidebar drawer"
            style={{ 
              color: 'var(--text-color-inverse)',
              minWidth: 'auto',
              padding: '8px'
            }}
          >
            <CloseIcon />
          </Button>
        </Flex>
      }
    >
      {view === "citations" && <Citations citations={citations} />}
    </SidePanel>
  );
}
