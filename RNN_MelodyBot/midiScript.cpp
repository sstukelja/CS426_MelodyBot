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
	//Take in long data argument as string object
	string longData(argv[1]);
	
	//Take in output file as string argument
	string shortData(argv[2]);

	//This vector will hold all info in longData in the form of strings
	vector<string> longDataStrings;
	
	//This will hold info in longData in the form of tokens
	vector<string> longDataTokens;

	//read in info from longData file///////////////////////////////////////
	fstream fin;
	fin.open( argv[1] );
	
	string input;  //this string will hold input temporarily
	
	//read in until through with file, store data as group of strings
	while ( getline (fin,input) )
    	{
        	longDataStrings.push_back(input);
    	}

	//close longData file
	fin.close();
	
	
	//Break group of longDataStrings into group of tokens for processing
	char * tempToken;
	for ( int i = 0; i < longDataStrings.size(); i++ )
	{
		char* tempString = strdup(longDataStrings[i].c_str());
		
		tempToken = strtok(tempString, " /,;:\n");
		while(tempToken != NULL)
		{
			longDataTokens.push_back(tempToken);
			tempToken = strtok(NULL, " /,;:\n");
		}
	}


	//open file to write shortData
	fstream fout;
	fout.open( shortData );

	int timingVar1 = 0;
	int timingVar2 = 0;
	int gap = 0;
	char noteChar = '0';

	for( int k = 0; k < longDataTokens.size(); k++ )
	{
		cout<< endl << longDataTokens[k] << endl;
	}

	//parse tokens to write shortData
	for ( unsigned int i = 0; i < longDataTokens.size(); i++ )
		{
			
			
			if ( longDataTokens[i] == "Note_on_c" && longDataTokens[i + 3] == "100" )
			{


				timingVar1 = atoi( (longDataTokens[i - 1]).c_str() );

				cout << endl << "Note Start: " << timingVar1;

				noteChar = '0' + (atoi( longDataTokens[i + 2].c_str() )) - 20;
				fout << noteChar;
			}
		
			
			else if ( longDataTokens[i] == "Note_on_c" && longDataTokens[i + 3] == "0" )
			{
				timingVar2 = atoi( (longDataTokens[i - 1]).c_str() );
				gap = ( abs(timingVar2 - timingVar1) / 20);

				cout << endl << "Note end: " << timingVar2;
				cout << endl << "gap: "<< gap << endl;

				for(int j = 0; j < gap; j++ )
				{
					fout << '.';
				}
				fout << endl;

			}

			/*else if ( longDataTokens[i] == "BREAK" )
			{
				fout << ".................................." << endl;

			}*/
	
		}//ends for loop
		
		//close log file
		fout.close();

}