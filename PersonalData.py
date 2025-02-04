def EmpNoAndName():
    # print('BOT  : Please give your ID number to check your record')
    # print("USER :", end=" ")
    while True:
        try:
            # emp_id = int(input())
            # cur.execute("SELECT `leave`, Employee_Name FROM employees WHERE Employee_Number = %s", (emp_id,))
            # result = cur.fetchone()
            result = [23, 'John Doe']
            if not result:
                return print("BOT  : Please register on the panel first.")
            leave_balance, name = result
            name = name.replace(',', '')
            response_index = 0 if leave_balance > 0 else 1
            print(intents['intents'][6]["responses"][response_index].format(bot_template, name, leave_balance))
        except ValueError:
            print("BOT  : Please enter a valid Employee ID")
            print("USER :", end=" ")
            continue
        break