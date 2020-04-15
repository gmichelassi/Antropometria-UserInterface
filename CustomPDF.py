from fpdf import FPDF


class CustomPDF(FPDF):
    def header(self):
        # Set up a logo
        self.image('./images/usp.png', 170, 7, 30)
        self.set_font('Arial', size=10)

        # Add an address
        self.cell(60)
        self.cell(0, 5, 'Sistema de auxílio ao diagnóstico de transtornos psiquiátricos', ln=1)
        # self.cell(100, border=1)
        # self.cell(0, 5, ' ', ln=1, border=1)

        self.ln(10)

    def footer(self):
        self.set_y(-10)

        self.set_font('Arial', 'I', 8)

        # Add a page number
        page = 'Page ' + str(self.page_no()) + '/{nb}'
        self.cell(0, 10, page, 0, 0, 'C')
