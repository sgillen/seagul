 //Write the integer x in its unique xLen-digit representation in base 256
 string str;
 str=cypher.get_str(256);

 //Print string figure by figure separated by space
for(int i=0;i<(int)str.length();i++){
    if(i%256==0)
        cout<<" "<<str[i];
    else
        cout<<str[i];
} 
