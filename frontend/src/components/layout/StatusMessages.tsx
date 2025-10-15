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

import { useNewCollectionStore } from "../../store/useNewCollectionStore";

export default function StatusMessages() {
  const { collectionName, error, uploadComplete } = useNewCollectionStore();

  return (
    <>
      {error && (
        <div className="mt-4 text-sm text-red-400 bg-red-900/30 p-3 rounded">{error}</div>
      )}

      {uploadComplete && (
        <div className="mt-4 p-3 bg-green-900/20 rounded text-sm text-green-300">
          Collection "{collectionName}" created successfully.
        </div>
      )}
    </>
  );
}
