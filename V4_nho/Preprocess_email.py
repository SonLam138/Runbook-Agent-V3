import re


def extract_clean_text_from_email(email):
    subject = email.get("subject", "")
    body = email.get("body", "")

    # Ghép subject + body
    text = subject + " " + body

    # Lowercase
    text = text.lower()

    # Remove signature
    signature_patterns = [
        r"thanks.*",
        r"regards.*",
        r"best regards.*",
        r"trân trọng.*",
        r"cảm ơn.*"
    ]

    for pattern in signature_patterns:
        text = re.sub(pattern, " ", text)

    # Remove special chars (giữ tiếng Việt)
    text = re.sub(
        r"[^a-zA-Z0-9áàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ\s]",
        " ",
        text
    )

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# email = {
#     "subject": "hỗ trợ kết nối Teams",
#     "body": "Tôi đăng nhập Teams thì bị báo lỗi access denied!"
# }

# clean_text = extract_clean_text_from_email(email)
# print(clean_text)