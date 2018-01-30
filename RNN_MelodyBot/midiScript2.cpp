#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>

using namespace std;

int main( int argc, char* argv[] )
{	
	//Take in og data argument as string object
	string ogData(argv[1]);

	//Take in destination file as string object
	string destinationFile(argv[2]);

	//This vector will hold all info in ogData in the form of strings
	vector<string> ogDataStrings;
	
	//This will hold info in ogData in the form of tokens
	vector<string> ogDataTokens;

	//read in info from ogData file///////////////////////////////////////
	fstream fin;
	fin.open( argv[1] );
	
	string input;  //this string will hold input temporarily
	
	//read in until through with file, store data as group of strings
	while ( getline (fin,input) )
    	{
        	ogDataStrings.push_back(input);
    	}

	//close ogData file
	fin.close();
	
	
	//Break group of ogDataStrings into group of tokens for processing
	char * tempToken;
	for ( int i = 0; i < ogDataStrings.size(); i++ )
	{
		char* tempString = strdup(ogDataStrings[i].c_str());
		
		tempToken = strtok(tempString, " /,;:\n");
		while(tempToken != NULL)
		{
			ogDataTokens.push_back(tempToken);
			tempToken = strtok(NULL, " /,;:\n");
		}
	}


	//open file to write shortData
	fstream fout;
	fout.open( argv[2] );


	int totalTime = 0;
	
	int noteInt = 0;

	int mod = ogDataTokens.size();



	for( int k = 0; k < ogDataTokens.size(); k++ )
	{
		cout<< endl << ogDataTokens[k] << endl;
	}

	//parse tokens to reconstruct longData
	for ( unsigned int i = 0; i < ogDataTokens.size(); i++ )
		{
			char letter = ogDataTokens[i][0];
			
			if ( letter != '.' )
			{

				noteInt  = ((int) letter) - 30;  //30 is perhaps wrong, 20?

				fout << "2, " << totalTime << ", " << "Note_on_c, " << "0, " << noteInt << ", 100" << endl;

				totalTime += (ogDataTokens[i].size() - 1) * 20;

				fout << "2, " << totalTime << ", " << "Note_on_c, 0, " << noteInt <<", 0" << endl;
			}
	
		}//ends for loop
		
		//close output file
		fout.close();

}