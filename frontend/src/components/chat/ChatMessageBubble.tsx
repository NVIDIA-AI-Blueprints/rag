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

import { useMemo } from "react";
import type { ChatMessage } from "../../types/chat";
import { useStreamingStore } from "../../store/useStreamingStore";
import { MessageContent } from "./MessageContent";
import { StreamingIndicator } from "./StreamingIndicator";
import { CitationButton } from "../citations/CitationButton";
import { 
  Block, 
  Flex, 
  Stack, 
  Panel
} from "@kui/react";

interface ChatMessageBubbleProps {
  msg: ChatMessage;
}

const MessageContainer = ({ 
  role, 
  isError = false,
  children 
}: { 
  role: "user" | "assistant"; 
  isError?: boolean;
  children: React.ReactNode;
}) => (
  <Flex justify={role === "user" ? "end" : "start"}>
    <Panel
      style={{
        maxWidth: '32rem',
        backgroundColor: role === "user" 
          ? 'var(--color-brand)' 
          : isError 
            ? 'var(--color-red-100)' 
            : 'var(--background-color-component-track-inverse)',
        color: role === "user" 
          ? 'black' 
          : isError
            ? 'var(--color-red-900)'
            : 'var(--text-color-accent-green)',
        border: isError ? '1px solid var(--color-red-300)' : undefined
      }}
    >
      {children}
    </Panel>
  </Flex>
);

const StreamingMessage = ({ content, isError = false }: { content: string; isError?: boolean }) => (
  <MessageContainer role="assistant" isError={isError}>
    <Flex align="center" gap="2">
      <MessageContent content={content} />
      {!content && <StreamingIndicator />}
    </Flex>
  </MessageContainer>
);

const RegularMessage = ({ msg }: { msg: ChatMessage }) => (
  <MessageContainer role={msg.role} isError={msg.is_error}>
    <Stack gap="2">
      <Block>
        <MessageContent content={msg.content ?? ""} />
      </Block>
      {msg.citations?.length && <CitationButton citations={msg.citations} />}
    </Stack>
  </MessageContainer>
);

export default function ChatMessageBubble({ msg }: ChatMessageBubbleProps) {
  const { isStreaming, streamingMessageId } = useStreamingStore();
  
  const isThisMessageStreaming = useMemo(() => 
    isStreaming && 
    msg.role === "assistant" && 
    streamingMessageId === msg.id, 
    [isStreaming, msg.role, msg.id, streamingMessageId]
  );

  if (isThisMessageStreaming) {
    return <StreamingMessage content={msg.content ?? ""} isError={msg.is_error} />;
  }

  return <RegularMessage msg={msg} />;
}
