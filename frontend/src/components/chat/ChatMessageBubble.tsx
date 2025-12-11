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

import { useMemo, useState } from "react";
import type { ChatMessage, MessageContent as MessageContentType } from "../../types/chat";
import { useStreamingStore } from "../../store/useStreamingStore";
import { MessageContent } from "./MessageContent";
import { StreamingIndicator } from "./StreamingIndicator";
import { CitationButton } from "../citations/CitationButton";
import { 
  Block, 
  Flex, 
  Stack, 
  Panel,
  Modal
} from "@kui/react";

/**
 * Extracts text content from multimodal content.
 */
const extractTextFromContent = (content: MessageContentType): string => {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .filter((item) => item.type === "text")
      .map((item) => (item as { type: "text"; text: string }).text)
      .join("\n");
  }
  return "";
};

/**
 * Extracts image URLs from multimodal content.
 */
const extractImagesFromContent = (content: MessageContentType): string[] => {
  if (typeof content === "string") {
    return [];
  }
  if (Array.isArray(content)) {
    return content
      .filter((item) => item.type === "image_url")
      .map((item) => (item as { type: "image_url"; image_url: { url: string } }).image_url.url);
  }
  return [];
};

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

const StreamingMessage = ({ content, isError = false }: { content: MessageContentType; isError?: boolean }) => {
  const textContent = extractTextFromContent(content);
  return (
    <MessageContainer role="assistant" isError={isError}>
      <Flex align="center" gap="2">
        <MessageContent content={textContent} />
        {!textContent && <StreamingIndicator />}
      </Flex>
    </MessageContainer>
  );
};

/**
 * Renders an image thumbnail in a message (clickable to open full size).
 */
const MessageImage = ({ src, onClick }: { src: string; onClick: () => void }) => (
  <button
    onClick={onClick}
    style={{
      borderRadius: "8px",
      overflow: "hidden",
      maxWidth: "200px",
      maxHeight: "200px",
      border: "none",
      padding: 0,
      cursor: "pointer",
      background: "transparent",
    }}
    title="Click to view full size"
  >
    <img
      src={src}
      alt="Attached image"
      style={{
        width: "100%",
        height: "100%",
        objectFit: "contain",
        display: "block",
      }}
    />
  </button>
);

/**
 * Modal to display full-size image.
 */
const ImageModal = ({ src, open, onClose }: { src: string | null; open: boolean; onClose: () => void }) => (
  <Modal
    open={open}
    onOpenChange={(isOpen) => !isOpen && onClose()}
    slotHeading=""
  >
    {src && (
      <img
        src={src}
        alt="Full size image"
        style={{
          maxWidth: "80vw",
          maxHeight: "80vh",
          objectFit: "contain",
        }}
      />
    )}
  </Modal>
);

const RegularMessage = ({ msg }: { msg: ChatMessage }) => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const textContent = extractTextFromContent(msg.content ?? "");
  const images = extractImagesFromContent(msg.content ?? "");
  
  return (
    <>
      <MessageContainer role={msg.role} isError={msg.is_error}>
        <Stack gap="2">
          {/* Render images first for user messages */}
          {images.length > 0 && (
            <Flex gap="2" wrap="wrap">
              {images.map((imgUrl, idx) => (
                <MessageImage 
                  key={idx} 
                  src={imgUrl} 
                  onClick={() => setSelectedImage(imgUrl)}
                />
              ))}
            </Flex>
          )}
          {/* Always render text content block to maintain structure */}
          <Block>
            <MessageContent content={textContent} />
          </Block>
          {msg.citations?.length && <CitationButton citations={msg.citations} />}
        </Stack>
      </MessageContainer>
      
      <ImageModal 
        src={selectedImage} 
        open={!!selectedImage} 
        onClose={() => setSelectedImage(null)} 
      />
    </>
  );
};

export default function ChatMessageBubble({ msg }: ChatMessageBubbleProps) {
  const { isStreaming, streamingMessageId } = useStreamingStore();
  
  const isThisMessageStreaming = useMemo(() => 
    isStreaming && 
    msg.role === "assistant" && 
    streamingMessageId === msg.id, 
    [isStreaming, msg.role, msg.id, streamingMessageId]
  );

  if (isThisMessageStreaming) {
    return <StreamingMessage content={msg.content} isError={msg.is_error} />;
  }

  return <RegularMessage msg={msg} />;
}
