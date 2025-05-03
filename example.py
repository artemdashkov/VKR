steps = "Navigate to registration page; Enter valid email and password; Click 'Register'"
steps_total = ""
for x in steps.split("; "):
    x = str(int(steps.split("; ").index(x))+1)+"."+ " " + x + "\n"
    steps_total += x
print(steps_total)