#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <vector> 


extern "C" HANDLE createAndConnectPipe(const char* pipeName, DWORD bufferSize) {
	HANDLE hPipe = CreateNamedPipe(
		pipeName,                 // Pipe name passed as argument
		PIPE_ACCESS_DUPLEX,        // Read/Write access
		PIPE_TYPE_BYTE |           // Byte-type pipe
		PIPE_READMODE_BYTE |       // Byte-read mode
		PIPE_WAIT,                 // Blocking mode
			1,                         // Max number of instances
		bufferSize,                // Output buffer size
		bufferSize,                // Input buffer size
		0,                         // Default timeout
		NULL);                     // Default security attributes

	if (hPipe == INVALID_HANDLE_VALUE) {
		std::cerr << "Failed to create named pipe.\n";
		return NULL;
	}

	std::cout << "Waiting for client connection...\n";
	BOOL connected = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);

	if (!connected) {
		std::cerr << "Failed to connect to the client.\n";
		CloseHandle(hPipe);
		return NULL;
	}

	return hPipe;
}


extern "C" bool CheckForRequest(HANDLE hPipe)
{
	DWORD bytesAvailable = 0;
	if (PeekNamedPipe(hPipe, NULL, 0, NULL, &bytesAvailable, NULL) && bytesAvailable > 0)
	{
		return true; // Data is available to read
	}
	return false; // No data available
}


extern "C" int handleClientRequests(
    HANDLE hPipe,
    short* data,
    short* dataB,
    double* corrMatrix,
    int segmentIndex,
    DWORD bytesToSend,
    int choice) // choice 0=data interleaved, 1=corrMatrix
{
    if (!CheckForRequest(hPipe))
        return 0;  // No request pending

    // 1️⃣ Read client request
    short request = -1;
    DWORD bytesRead = 0;
    BOOL success = ReadFile(hPipe, &request, sizeof(request), &bytesRead, NULL);
    if (!success || bytesRead != sizeof(request)) {
        std::cerr << "[Error] Failed to read request from client.\n";
        return 1;
    }

    DWORD bytesWritten = 0;

    // 2️⃣ Interleaved data mode (choice == 0)
    if (choice == 0)
    {
        // Number of samples per segment
        size_t samplesPerSegment = bytesToSend / sizeof(short);

        // Calculate start offsets
        short* segA = data + segmentIndex * samplesPerSegment;
        short* segB = dataB + segmentIndex * samplesPerSegment;

        // Allocate temporary interleaved buffer (stack or heap)
        std::vector<short> interleaved(samplesPerSegment * 2);

        // Interleave: A0,B0,A1,B1,...
        for (size_t i = 0; i < samplesPerSegment; ++i)
        {
            interleaved[2 * i] = segA[i];
            interleaved[2 * i + 1] = segB[i];
        }

        DWORD totalBytes = static_cast<DWORD>(interleaved.size() * sizeof(short));
        success = WriteFile(hPipe, interleaved.data(), totalBytes, &bytesWritten, NULL);

        if (!success || bytesWritten != totalBytes)
        {
            std::cerr << "[Error] Failed to send interleaved data.\n";
            return 1;
        }
    }

    // 3️⃣ Correlation matrix mode (choice == 1)
    else if (choice == 1)
    {
        double* segC = corrMatrix + (segmentIndex * (bytesToSend / sizeof(double)));
        success = WriteFile(hPipe, segC, bytesToSend, &bytesWritten, NULL);

        if (!success || bytesWritten != bytesToSend)
        {
            std::cerr << "[Error] Failed to send correlation matrix.\n";
            return 1;
        }
    }

    else
    {
        std::cerr << "[Warning] Unknown choice parameter.\n";
        return 1;
    }

    return 2;  // Successfully sent
}

