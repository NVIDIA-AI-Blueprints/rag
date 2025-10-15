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

import { useCallback } from "react";
import { useChatStore } from "../../store/useChatStore";
import { useStreamingStore } from "../../store/useStreamingStore";
import { useTextareaResize } from "../../hooks/useTextareaResize";
import { useMessageSubmit } from "../../hooks/useMessageSubmit";
import { TextArea } from "@kui/react";

interface MessageTextareaProps {
  placeholder?: string;
}

export const MessageTextarea = ({ 
  placeholder = "Ask a question about your documents..." 
}: MessageTextareaProps) => {
  const { input, setInput } = useChatStore();
  const { isStreaming } = useStreamingStore();
  const { handleInput, getTextareaStyle } = useTextareaResize();
  const { handleSubmit, canSubmit } = useMessageSubmit();

  // Type-safe keyboard event handler for TextArea
  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && canSubmit) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit, canSubmit]);

  // Type-safe change event handler for TextArea
  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  }, [setInput]);

  // Type-safe bridge functions: convert KUI's HTMLDivElement events to HTMLTextAreaElement events
  const bridgeKeyboardEvent = (handler: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void) => {
    return (kuiEvent: React.KeyboardEvent<HTMLDivElement>) => {
      const target = kuiEvent.target;
      if (target && target instanceof HTMLTextAreaElement) {
        // Create a proxy event that preserves all methods and properties
        const textareaEvent = Object.create(kuiEvent);
        textareaEvent.target = target;
        textareaEvent.currentTarget = target;
        handler(textareaEvent as React.KeyboardEvent<HTMLTextAreaElement>);
      }
    };
  };

  const bridgeChangeEvent = (handler: (event: React.ChangeEvent<HTMLTextAreaElement>) => void) => {
    return (kuiEvent: React.FormEvent<HTMLDivElement>) => {
      const target = kuiEvent.target;
      if (target && target instanceof HTMLTextAreaElement && 'value' in target) {
        // Create a proxy event that preserves all methods and properties
        const textareaEvent = Object.create(kuiEvent);
        textareaEvent.target = target;
        textareaEvent.currentTarget = target;
        handler(textareaEvent as React.ChangeEvent<HTMLTextAreaElement>);
      }
    };
  };

  const bridgeFormEvent = (handler: (event: React.FormEvent<HTMLTextAreaElement>) => void) => {
    return (kuiEvent: React.FormEvent<HTMLDivElement>) => {
      const target = kuiEvent.target;
      if (target && target instanceof HTMLTextAreaElement) {
        // Create a proxy event that preserves all methods and properties
        const textareaEvent = Object.create(kuiEvent);
        textareaEvent.target = target;
        textareaEvent.currentTarget = target;
        handler(textareaEvent as React.FormEvent<HTMLTextAreaElement>);
      }
    };
  };

  return (
    <TextArea
      placeholder={placeholder}
      rows={1}
      value={input}
      onChange={bridgeChangeEvent(handleChange)}
      onKeyDown={bridgeKeyboardEvent(handleKeyDown)}
      onInput={bridgeFormEvent(handleInput)}
      disabled={isStreaming}
      size="medium"
      resizeable="auto"
      style={{
        width: '100%',
        backgroundColor: 'var(--background-color-surface)',
      }}
      attributes={{
        TextAreaElement: {
          style: {
            paddingLeft: '78px',
            paddingRight: '56px',
            paddingTop: '5px',
            lineHeight: '22px',
            border: 'none',
            outline: 'none',
            ...getTextareaStyle(),
            height: '32px',
            minHeight: '32px'
          }
        }
      }}
    />
  );
}; 