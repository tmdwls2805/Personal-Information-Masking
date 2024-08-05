import re


class TokenMasker:
    def __init__(self, token):
        self.token = token

    def examine(self):
        result = ""
        tag = ""

        if self.maskingRRN() is not None:
            result = self.maskingRRN()
            tag = 'RRN'
        else:
            if self.maskingTEL() is not None:
                result = self.maskingTEL()
                tag = 'TEL'
            else:
                if self.maskingDATE() is not None:
                    result = self.maskingDATE()
                    tag = 'DATE'
                else:
                    if self.maskingEMAIL() is not None:
                        result = self.maskingEMAIL()
                        tag = 'MAIL'
                    else:
                        result = None

        return result, tag

    # 주민등록번호
    def maskingRRN(self):
        rrn_regex = re.compile(r'([0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[1,2][0-9]|3[0,1])-([1-4]{1})([0-9]{6})|'
                               r'([0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[1,2][0-9]|3[0,1])([1-4]{1})([0-9]{6})|'
                               r'([0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[1,2][0-9]|3[0,1])|'
                               r'([1-4]{1})([0-9]{6})')

        if rrn_regex.match(self.token):
            rrn = rrn_regex.match(self.token).group()
            f_token = rrn_regex.match(self.token).string
            rrn_length = len(rrn)

            rem = f_token[rrn_length:]

            if len(rrn) > 13:
                result = (rrn_regex.sub("\g<1>****-\g<4>******" + rem, rrn), rrn)
            elif len(rrn) == 13:
                result = (rrn_regex.sub("\g<6>****\g<9>******" + rem, rrn), rrn)
            elif len(rrn) == 7:
                result = (rrn_regex.sub("\g<14>******" + rem, rrn), rrn)
            else:
                result = (rrn_regex.sub("\g<11>****" + rem, rrn), rrn)
        else:
            result = None

        return result

    # 전화번호
    def maskingTEL(self):
        tel_regex = re.compile(r'(\d{2,3})([-.)])(\d{3,4})([-.])(\d{4})|'
                               r'([0-9]{2,3})([0-9]{7,8})')

        if tel_regex.match(self.token):
            tel = tel_regex.match(self.token).group()
            f_token = tel_regex.match(self.token).string
            tel_length = len(tel)

            rem = f_token[tel_length:]
            sym = any(s in tel for s in '-.)')

            if len(tel) > 12:
                result = (tel_regex.sub("\g<1>\g<2>****\g<4>****" + rem, tel), tel)
            elif sym and (len(tel) == 12 or len(tel) == 11):
                result = (tel_regex.sub("\g<1>\g<2>***\g<4>****" + rem, tel), tel)
            elif sym is not True and len(tel) == 11:
                result = (tel_regex.sub("\g<6>********" + rem, tel), tel)
            elif sym is not True and (len(tel) == 9 or len(tel) == 10):
                result = (tel_regex.sub("\g<6>*******" + rem, tel), tel)
            else:
                result = None
        else:
            result = None

        return result

    # 날짜
    def maskingDATE(self):
        date_regex = re.compile(r'([0-9]{2})[0-9]{2}([-.])[0,1]{1}[0-9]{1}([-.])[0-3][0-9]{1}|'
                                r'([0-9]{2})([-.])[0,1]{1}[0-9]{1}([-.])[0-3][0-9]{1}|'
                                r'[0-9]{4}([-.])[0,1]{1}[0-9]{1}|'
                                r'([0-2]{2})[0-9]{2}[0,1]{1}[0-9]{1}[0-3]{1}[0-9]{1}|'
                                r'[0-9]{4}([년])[0-1]{1}[0-9]{1}([월])[0-3]{1}[0-9]{1}([일])|'
                                r'[0-9]{1,4}([년월일])|'
                                r'[0-9]{1,3}(개월)|'
                                r'[0-9]{1,2}(달)')

        if date_regex.match(self.token):
            date = date_regex.match(self.token).group()

            f_token = date_regex.match(self.token).string
            date_length = len(date)

            rem = f_token[date_length:]

            sym = any(s in date for s in '-.')
            sym2 = any(s in date for s in '년월일')

            if sym and len(date) > 9:
                result = (date_regex.sub("\g<1>**\g<2>**\g<3>**" + rem, date), date)
                # print(date)
            elif sym and len(date) == 8:
                result = (date_regex.sub("\g<4>\g<5>**\g<6>**" + rem, date), date)
                # print(date)
            elif sym and len(date) == 7:
                result = (date_regex.sub("****\g<7>**" + rem, date), date)
            elif sym is not True and len(date) == 8:
                result = (date_regex.sub("\g<8>******" + rem, date), date)
                # print(date)
            elif sym2 and len(date) >= 8:
                result = (date_regex.sub("****\g<9>**\g<10>**\g<11>" + rem, date), date)
                # print(date)
            elif '개월' in date:
                # print(date)
                result = (date_regex.sub("**\g<13>" + rem, date), date)
            elif '달' in date:
                # print(date)
                result = (date_regex.sub("*\g<14>" + rem, date), date)
            elif sym2 and len(date) >= 2:
                result = (date_regex.sub("**\g<12>" + rem, date), date)
                # print(date)
            else:
                result = None
        else:
            result = None

        return result

    # 이메일
    def maskingEMAIL(self):
        email_regex = re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,4})")

        if email_regex.match(self.token):
            email = email_regex.match(self.token).group()
            f_token = email_regex.match(self.token).string

            email_length = len(email)
            rem = f_token[email_length:]
            v_length = len(email_regex.sub("\g<1>", email))
            result = (email_regex.sub("*" * v_length + "@" + "\g<2>" + rem, email), email)
        else:
            result = None

        return result