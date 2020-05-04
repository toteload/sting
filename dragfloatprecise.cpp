bool DragFloatPrecise(const char* label, float* v, float v_speed = 1.0f) {
    // Some functions I came across while working on this widget that might be
    // handy to use at some point. I am still learning :)
    //
    // CalcTextSize, might be a handy function to use
    // ItemSize() ??
    // FindBestWindowPosForPopup
    //
    // TODO and issues 
    // - Calculate the width of the popup window to fit better, right now it is a hardcoded value >:D
    // - When the popup is close to the top or bottom of the screen part is not visible/usable, which might be annoying. 
    // - Currently the float value drawn in the InputFloat box lags behind 1 frame.

    bool value_changed = false;

    ImGuiContext& g = *GImGui;

    ImGui::PushID(label);
 
    ImGuiStorage* storage = ImGui::GetStateStorage();
    const ImGuiID store_id = ImGui::GetID("selection");
    const ImGuiID init_value_id = ImGui::GetID("initval");
    const ImGuiID prev_value_id = ImGui::GetID("prevval");
                                                     
    value_changed = ImGui::InputFloat(label, v, 0.0f, 0.0f, "%.4f");

    const ImGuiMouseButton popup_button = ImGuiMouseButton_Middle;

    if (ImGui::IsItemClicked(popup_button)) {
        ImGui::OpenPopup("f32select");        
        storage->SetFloat(init_value_id, *v);
        storage->SetFloat(prev_value_id, *v);
    }

    const i32 item_count   = 7;
    const char* option_text[]  = { "100", "10", "1", "0.1", "0.01", "0.001", "0.0001", };
    const i32 option_textlen[] = { 3, 2, 1, 3, 4, 5, 6, };
    const f32 option_values[]  = { 100.0f, 10.0f, 1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, };
 
    const f32 entry_height = 2 * ImGui::GetTextLineHeightWithSpacing();
    const f32 popup_width  = 100.0f;
    const f32 popup_height = item_count * entry_height;
    
    if (ImGui::IsPopupOpen("f32select")) {
        // This might be a bit hacky, I am not sure, but it works!
        // The IsPopupOpen check also accesses the OpenPopupStack in the same way.
        const ImVec2 openmouse_pos = g.OpenPopupStack[g.BeginPopupStack.Size].OpenMousePos;
        ImGui::SetNextWindowPos(ImVec2(openmouse_pos.x - popup_width / 2.0f, openmouse_pos.y - popup_height / 2.0f));
        ImGui::SetNextWindowSize(ImVec2(popup_width, popup_height));
        ImGui::SetNextWindowBgAlpha(1.0f);
    }
    
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    if (ImGui::BeginPopup("f32select")) {
        ImGuiWindow* window = g.CurrentWindow;
        ImDrawList* draw_list = window->DrawList;

        const ImVec2 popup_size = ImGui::GetWindowSize();
        const ImVec2 popup_pos  = ImGui::GetWindowPos();

        const ImVec2 popup_min = popup_pos;
        const ImVec2 popup_max = ImVec2(popup_min.x + popup_size.x, popup_min.y + popup_size.y);

        const ImGuiStyle& style = g.Style;
        const ImVec2 p = window->DC.CursorPos;

        f32 offset;
        if (g.IO.MousePos.x >= popup_min.x && g.IO.MousePos.x < popup_max.x) {
            offset = 0.0f;
        } else {
            if (g.IO.MousePos.x > popup_min.x) {
                offset = g.IO.MousePos.x - popup_max.x;
            } else {
                offset = g.IO.MousePos.x - popup_min.x;
            }
        }

        offset *= v_speed;

        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);

        bool inside_popup = false;
        i32 selected = -1;

        // Find the currently selected box
        for (i32 i = 0; i < item_count; i++) {
            if (g.IO.MousePos.x >= p.x                    && g.IO.MousePos.x < p.x + popup_width && 
                g.IO.MousePos.y >= p.y + i * entry_height && g.IO.MousePos.y < p.y + (i + 1) * entry_height) 
            {
                selected = i;
                inside_popup = true;
                break;
            }
        }

        // If the mouse is not hovering over a selection, then we must have the
        // option saved, so we retrieve it.
        if (selected == -1) {
            selected = storage->GetInt(store_id, -1);
        }

        IM_ASSERT(selected != -1);

        const f32 init_value = storage->GetFloat(init_value_id);
        const f32 prev_value = storage->GetFloat(prev_value_id);
        const f32 current_value = init_value + offset * option_values[selected];
        *v = current_value;
        value_changed = current_value != prev_value;
        storage->SetFloat(prev_value_id, current_value);

        // Save our selected option
        storage->SetInt(store_id, selected);

        // Highlight the currently selected option
        draw_list->AddRectFilled(ImVec2(p.x,               p.y + selected * entry_height), 
                                 ImVec2(p.x + popup_width, p.y + (selected + 1) * entry_height), 
                                 ImGui::GetColorU32(ImGuiCol_FrameBgActive));

        // Draw separators
        for (i32 i = 1; i < item_count; i++) {
            draw_list->AddLine(ImVec2(p.x, p.y + i * entry_height),
                               ImVec2(p.x + popup_width, p.y + i * entry_height),
                               ImGui::GetColorU32(ImGuiCol_Separator));
        }

        // Draw the labels for all the options
        for (i32 i = 0; i < item_count; i++) {
            const ImVec2 frame_padding = style.FramePadding;

            const f32 align_y = (i == selected) ? 0.0f : 0.5f;

            ImVec2 text_size = ImGui::CalcTextSize(option_text[i], option_text[i] + option_textlen[i]);
            ImVec2 padding = ImVec2(popup_width  - 2.0f * frame_padding.x - text_size.x,
                                    entry_height - 2.0f * frame_padding.y - text_size.y);

            draw_list->AddText(ImVec2(p.x + frame_padding.x + padding.x * 0.5f,
                                      p.y + frame_padding.y + padding.y * align_y + i * entry_height),
                               ImGui::GetColorU32(ImGuiCol_Text),
                               option_text[i], option_text[i] + option_textlen[i]);

            // Draw the value in the highlighted option box
            if (i == selected) {
                 char buf[64];
                 const i32 buflen = ImGui::DataTypeFormatString(buf, 64, ImGuiDataType_Float, v, "%.4f");
                 text_size = ImGui::CalcTextSize(buf, buf + buflen);
                 padding = ImVec2(popup_width  - 2.0f * frame_padding.x - text_size.x,
                                  entry_height - 2.0f * frame_padding.y - text_size.y);
                   
                 draw_list->AddText(ImVec2(p.x + frame_padding.x + padding.x * 0.5f,
                                           p.y + frame_padding.y + padding.y * 1.0f + i * entry_height),
                                    ImGui::GetColorU32(ImGuiCol_Text),
                                    buf, buf + buflen);
            }
        }

        if (!ImGui::IsMouseDown(popup_button)) { 
            ImGui::CloseCurrentPopup(); 
        }

        ImGui::EndPopup();
    }

    ImGui::PopStyleVar();
    ImGui::PopID();

    return value_changed;
}
