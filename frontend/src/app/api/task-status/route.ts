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

import { NextResponse, NextRequest } from "next/server";
import { createErrorResponse } from "../utils/api-utils";
import { API_CONFIG, buildQueryUrl } from "@/app/config/api";

// GET /task-status
export async function GET(request: NextRequest) {
  try {
    const taskId = request.nextUrl.searchParams.get("task_id");
    
    if (!taskId) {
      return NextResponse.json(
        { error: "Missing task_id parameter" },
        { status: 400 }
      );
    }

    const url = buildQueryUrl(
      `${API_CONFIG.VDB.BASE_URL}${API_CONFIG.VDB.ENDPOINTS.TASK_STATUS}`,
      { task_id: taskId }
    );

    console.log(`Checking status for task: ${taskId}`);
    
    const response = await fetch(url, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      console.error(`Failed to get task status: ${response.statusText}`);
      return NextResponse.json(
        { error: `Failed to get task status: ${response.statusText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error checking task status:", error);
    return createErrorResponse(error);
  }
} 