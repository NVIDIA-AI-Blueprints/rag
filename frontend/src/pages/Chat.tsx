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

import { useRef, useEffect } from "react";
import { useChatStore } from "../store/useChatStore";
import MessageInput from "../components/chat/MessageInput";
import ChatMessageBubble from "../components/chat/ChatMessageBubble";
import CollectionList from "../components/collections/CollectionList";
import SidebarDrawer from "../components/drawer/SidebarDrawer";
import { Flex, Grid, GridItem, Stack } from "@kui/react";

/**
 * Main chat page component that displays the conversation interface.
 * 
 * Renders chat messages, message input, collection list, and sidebar drawer.
 * Automatically scrolls to show the latest message.
 * 
 * @returns The chat page component
 */
export default function Chat() {
  const { messages } = useChatStore();

  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <>
      <Grid 
        cols={12}
        style={{ 
          height: "calc(100vh - 48px)",
          backgroundColor: 'var(--background-color-surface-base)'
        }}
      >
        {/* Sidebar GridItem */}
        <GridItem cols={3} style={{ height: 'calc(100vh - 48px)' }}>
          <CollectionList />
        </GridItem>

        <GridItem cols={9}>
          <Flex direction="col" justify="between" style={{ height: 'calc(100vh - 48px)' }}>
            <Stack gap="density-lg" padding="density-lg" style={{ overflow: 'auto' }}>
              {messages.map((msg) => (
                <ChatMessageBubble key={msg.id} msg={msg} />
              ))}
              <div ref={bottomRef} />
            </Stack>
            <MessageInput />
          </Flex>
        </GridItem>
      </Grid>
      
      <SidebarDrawer />
    </>
  );
}
