print( "%-20s\t\t%-20s\t\t%-20s\n"%("name","phone","email")+"-" * 50)
for info in card_list:
    print("%-20s\t\t%-20s\t\t%-20s"%(info["name"],info["phone"],info["email"]))
