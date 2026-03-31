/**
 * @file export.h
 * @brief JSON export interface for CB_DistList_t operation records.
 *
 * Provides CB_dlist_export_json, which serializes a gathered CB_DistList_t
 * to a JSON file for offline analysis and visualization of collective
 * algorithm performance data.
 *
 * @author DanieleSavino <savino.2140356@studenti.uniroma1.it>
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Daniele - Sapienza Università di Roma
 */
#pragma once
#include "CollBench/dist_list.h"
#include "CollBench/errors.h"

/**
 * @brief Serializes a CB_DistList_t to a JSON file.
 *        Typically called on the root rank after CB_dlist_gather.
 *        Creates or overwrites the file at path.
 *        Records with t_end_ns == 0 are considered invalid and cause
 *        CB_ERR_INVALID_ARG to be returned.
 * @param list The gathered operation list to export.
 * @param path Path to the output JSON file.
 * @return CB_SUCCESS, CB_ERR_NULLPTR, CB_ERR_IO, or CB_ERR_INVALID_ARG.
 */
CB_Error_t CB_dlist_export_json(const CB_DistList_t *list, const char *path);
