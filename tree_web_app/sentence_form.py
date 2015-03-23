from web import form

the_form = form.Form(
    form.Textbox("text", description="Please Enter Some Text"),
    form.Button("submit", type="submit", description="Parse Text")
)
