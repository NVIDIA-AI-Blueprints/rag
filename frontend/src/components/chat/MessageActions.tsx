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

import { useSendMessage } from "../../api/useSendMessage";
import { useStreamingStore } from "../../store/useStreamingStore";
import { useMessageSubmit } from "../../hooks/useMessageSubmit";
import { Button, Block, Flex, Spinner } from "@kui/react";

const StopIcon = () => (
  <div style={{ width: '8px', height: '8px', backgroundColor: 'currentColor', borderRadius: '2px' }} />
);

const SendIcon = () => (
  <svg style={{ width: '14px', height: '14px' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
);

const StopButton = () => {
  const { stopStream } = useSendMessage();

  return (
    <Button
      kind="primary"
      color="danger"
      size="small"
      onClick={stopStream}
    >
      <Flex align="center" gap="density-xs">
        <StopIcon />
        Stop
      </Flex>
    </Button>
  );
};

const SendButton = () => {
  const { handleSubmit, canSubmit, isHealthLoading, shouldDisableHealthFeatures } = useMessageSubmit();

  const getButtonContent = () => {
    if (isHealthLoading) {
      return (
        <Flex align="center" gap="density-xs">
          <Spinner size="small" aria-label="Loading system configuration" />
        </Flex>
      );
    }
    return <SendIcon />;
  };

  const getButtonTitle = () => {
    if (isHealthLoading) {
      return "Loading system configuration...";
    }
    if (shouldDisableHealthFeatures) {
      return "System configuration unavailable";
    }
    return "Send message";
  };

  return (
    <Button
      kind="primary"
      color="brand"
      size="small"
      onClick={handleSubmit}
      disabled={!canSubmit}
      title={getButtonTitle()}
    >
      {getButtonContent()}
    </Button>
  );
};

export const MessageActions = () => {
  const { isStreaming } = useStreamingStore();

  return (
    <Block style={{ position: 'absolute', right: '12px', top: '50%', transform: 'translateY(-50%)' }}>
      <Flex align="center" gap="density-sm">
        {isStreaming ? <StopButton /> : <SendButton />}
      </Flex>
    </Block>
  );
}; 