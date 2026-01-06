import inflect
import re

p = inflect.engine()

def normalize_text(text: str) -> str:
    currency_map = {
        "$": "dollars",
        "£": "pounds",
        "€": "euros",
        "¥": "yen",
        "Rs.": "Rupees",
        "Rs": "Rupees",
        "₹": "Rupees"
    }

    # --- Currency replacement including Rs / Rs. ---
    def money_replacer(match):
        currency_symbol, num = match.groups()
        num_without_commas = num.replace(',', '')

        if '.' in num_without_commas:
            whole, decimal = num_without_commas.split('.')
            w_words = p.number_to_words(int(whole))
            d_words = p.number_to_words(int(decimal))
            return f"{w_words} {currency_map.get(currency_symbol, 'currency')} and {d_words} cents"
        else:
            n_words = p.number_to_words(int(num_without_commas))
            return f"{n_words} {currency_map.get(currency_symbol, 'currency')}"

    text = re.sub(r"(Rs\.?|₹|[$£€¥])\s?(\d+(?:,\d{3})*(?:\.\d{2})?)", money_replacer, text)

    def phone_replacer(match):
        return ", ".join(" ".join(p.number_to_words(int(d)) for d in group) for group in match.groups())

    text = re.sub(r"(\d{3})-(\d{3})-(\d{4})", phone_replacer, text)

    def mobile_replacer(match):
        digits = match.group()
        return " ".join(p.number_to_words(int(d)) for d in digits)

    text = re.sub(r"\b\d{10}\b", mobile_replacer, text)

    return text


# Example usage
print(normalize_text("$1,000"))
print(normalize_text("Rs. 1000"))
print(normalize_text("₹1000"))
print(normalize_text("Rs.1000"))
print(normalize_text("$1,234.56"))
print(normalize_text("555-555-5555"))
print(normalize_text("9617373847"))
