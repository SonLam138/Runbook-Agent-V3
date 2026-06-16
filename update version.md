Tại V2 :
User → embed → vector search → return runbook
thành phần :
-Embedding (BGE)
-Vector search (NumPy)
-Threshold
-Cache đơn giản

Tại V3mini :
User --> Agent Runtime (LLM + state) --> Tool (Search_RB/Search_RB_topK) --> Session memory + semantic memory --> trả kết quả + retry nếu fail

Các kỹ thuật bổ sung :
1. Agent Decision (LLM control flow) gồm các hàm :
    - decide_initial()
    - decide_with_state()
ý nghĩa : không chỉ trả lời, mà còn quyết định:
    - search hay ask_more
    - hỏi cái gì tiếp theo

2. Agent multi-turn :
 - Conversation State (Session Memory)
    + get_session(session_id)
    + update_session(session_id, state)
    + reset_session(session_id)
 --> agent có thể nhớ user đang nói gì ở lượt trước
 - Multi-turn Loop (Agent Runtime Loop)
    + run_agent(session_id, user_input)
 -->  
 - Slot Filling (Thu thập thông tin thiếu)
    + fill_pending_slot(state, user_input)

2. Agent multi-turn :
 - issue : query mơ hồ, không tìm được RB, agent không hỏi tiếp, nếu user query lại --> query mới, độc lập
 - Giải pháp cho agent : ask_more → user trả lời → tiếp tục xử lý
 - Code :
  + run_agent(session_id, user_input)
--> agent đã hỏi tiếp user nếu query mơ hồ

3. Session Memory
 - issue : sau khi multi-turn, phát sinh agent không nhớ câu hỏi trước của user
 - giải pháp cho agent : Tạo bộ nhớ cho agent hay Session-based state management
 - Code :
  + thêm file session_store.py    
        get_session()
        update_session()
        reset_session()
  + agent.py
        state = get_session(session_id)
        append_history(state, ...)
--> agent đã nhớ câu trả lời của user qua từng turn

4. Slot Filling
 - issue : sau khi có bộ nhớ, agent có thể hỏi lại user sau query mơ hồ, user trả lời câu hỏi đó, agent đã có thể nhớ lần trả lời thứ 2... nhưng agent không biết thiếu gì, làm thế nào để tổng hợp các lần hỏi thành 1 query hoàn chỉnh cho việc search RB
 - Giải pháp cho agent : tạo cho agent biết thiếu gì để hỏi cái đó
 - Code :
    + thêm slot state : thiết lập các điều kiện để agent biết đang thiếu gì
        tate["slots"] = {
        "issue_type": None,
        "service": None,
        "error_message": None
        }
    + agent.py : agent sẽ hỏi và điền thêm vào slot còn thiếu
        fill_pending_slot()
        pending_slot
--> Agent có thể hỏi nhiều lần, nhớ lần hỏi trước, biết câu trả lời còn thiếu gì để hỏi và điền

5. Clarification Loop
 - issue : sau khi có slot, agent đã hỏi thêm và điền thêm giá trị nhưng chỉ hỏi 1 lần
 - giải pháp cho agent : sau khi điền, nếu chưa đủ thì hỏi tiếp
 - Code :
    if state["mode"] == "clarifying":
--> Agent có thể hỏi nhiều lần, nhớ lần hỏi trước, biết câu trả lời còn thiếu gì để hỏi và điền đến khi đủ điều kiện đã đặt ra

 - Nâng cao : 
    + trước : sau khi Clarification Loop, agent chỉ nối chuỗi các query : "turn1 + turn2 + turn3" 
    VD : "bị lỗi" "exchange online" "quên mật khẩu"
    + sau : thêm semantic memory trong agent,py
     semantic_query = build_semantic_query(state)
 --> hiểu ngữ nghĩa để build search_RB query "con người" hơn

6. Failure Feedback
 - issue : agent quyết định trả RB A, nhưng user nói (feedback) "vẫn lỗi" --> agent không hiểu vẫn trong luồng câu hỏi hiện tại nên thực hiện hỏi lại từ đầu, tạo cảm giác "Agent bị ngu"
 - Giải pháp cho agent : hiểu đâu là failure feedback tức là biết RB vừa đưa ra không đúng, cần phải xử lý tiếp theo.
 - Code : thêm vào agent.py
    is_failure_feedback(user_input)
    FAILURE_SIGNALS = [...] : các từ khóa để xác định là Fail
--> Agent đã biết solution trước không đúng

7. Failure Handling (retry logic)
 - issue : agent đã biết solution trước không đúng, nhưng không biết làm gì tiếp theo
 - Giải pháp cho agent : xây dựng logic xử lý, vd như tiếp tục hỏi, chuyển tiếp yêu cầu... trong agent này chọn giải pháp handle là đưa ra thêm RB khác
 - Code : thêm vào agent.py
    + handle_failure_retry(session_id, state, user_input)
    + Do chọn logic retry là đưa thêm RB nên cần xử lý lại tool search_RB để trả ra hơn 1 RB
    + Thêm hoặc sửa hàm Search_runbook trong RB_query_v2_cache.py : search_RB_topk(query, exclude_titles, topk)
--> Agent đã biết nếu A fail --> thử B --> thử C

8. semantic cache theo session
 - issue : agent biết thử A, thử B, thử C nhưng có thể đưa ra các RB trùng nhau
 - Giải pháp cho agent : phải nhớ đã search RB nào, trả lời RB nào để thử RB khác
 - Code :
    + thêm vào state    
        semantic_query
        tried_runbooks
        last_runbook
        semantic_cache
    + thêm vào agent.py        
        remember_success()
        get_latest_semantic_query()

9. Turning failure feedback
 - hiện tại : fail detection đang dựa vào signal có sẵn, dễ bị phát hiện thiếu
 - Giải pháp : Hybrid LLM 
 - Code :
    def rule_detect_failure(user_input: str)
    def llm_detect_failure(user_input: str, state: dict)
    sửa : def is_failure_feedback(user_input: str, state: dict)
    gắn vào run_agent() :
        if is_failure_feedback(user_input, state):
            print("⚠️ FAILURE FEEDBACK DETECTED")
            return handle_failure_retry(session_id, state, user_input)
--> Sau khi turning : agent nhận biết failure qua rule trước, nếu không có trong rule thì để LLM phân tích và trả kết quả

10. thêm 2 layer : service resolution engine và Knowledge  Grounding Layer
 - issue : 
    LLM dựa vào general knowledge
    Không hiểu môi trường IT nội bộ
    Dẫn đến khi cần hỏi clarify thì hỏi chung chung, không sát thực tế PV
 - Giải pháp : thêm 2 layer như trên
  + Service Resolution Engine : xác định service ngay từ đầu, dựa vào cả rule và semantic
   Hàm chính : resolve_service(query, state)
    Hàm con :
     rule_match_service(query) : match nhanh bằng rule(trong file service catalog.json) : khi rule không match, dùng embed + faiss để hiểu ngữ nghĩa 
  --> LLM không đoán service mà làm việc trong service
  + Knowledge Grounding Layer : Kiểm soát cách LLM suy nghĩ và hỏi
   Các hàm :
    get_clarify_knowledge(service) : trả về rule để LLM hỏi
    get_decision_knowledge(service) : trả về rule để LLM quyết định action : search / ask more
   Các hàm được inject knowledge
    generate_clarify_message_llm
    decide_with_candidates
   Các hàm loại bỏ :
    build_semantic_query
--> LLM hỏi sát thực tế hơn (knowledge base càng nhiều, càng sát thì LLM hỏi càng đúng với nghiệp vụ tại PV, có promt không hỏi lại thông tin user đã cung cấp)