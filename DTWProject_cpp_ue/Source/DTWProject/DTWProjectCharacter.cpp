// Copyright 1998-2018 Epic Games, Inc. All Rights Reserved.

#include "DTWProjectCharacter.h"
#include "HeadMountedDisplayFunctionLibrary.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Components/InputComponent.h"
#include "GameFramework/CharacterMovementComponent.h"
#include "GameFramework/Controller.h"
#include "GameFramework/SpringArmComponent.h"
#include "Runtime/Core/Public/Misc/FileHelper.h"
#include "Runtime/Core/Public/HAL/PlatformFilemanager.h"

//////////////////////////////////////////////////////////////////////////
// ADTWProjectCharacter

ADTWProjectCharacter::ADTWProjectCharacter()
{
	// Set size for collision capsule
	GetCapsuleComponent()->InitCapsuleSize(42.f, 96.0f);

	// set our turn rates for input
	BaseTurnRate = 45.f;
	BaseLookUpRate = 45.f;

	// Don't rotate when the controller rotates. Let that just affect the camera.
	bUseControllerRotationPitch = false;
	bUseControllerRotationYaw = false;
	bUseControllerRotationRoll = false;

	// Configure character movement
	GetCharacterMovement()->bOrientRotationToMovement = true; // Character moves in the direction of input...	
	GetCharacterMovement()->RotationRate = FRotator(0.0f, 540.0f, 0.0f); // ...at this rotation rate
	GetCharacterMovement()->JumpZVelocity = 600.f;
	GetCharacterMovement()->AirControl = 0.2f;

	// Create a camera boom (pulls in towards the player if there is a collision)
	CameraBoom = CreateDefaultSubobject<USpringArmComponent>(TEXT("CameraBoom"));
	CameraBoom->SetupAttachment(RootComponent);
	CameraBoom->TargetArmLength = 300.0f; // The camera follows at this distance behind the character	
	CameraBoom->bUsePawnControlRotation = true; // Rotate the arm based on the controller

	// Create a follow camera
	FollowCamera = CreateDefaultSubobject<UCameraComponent>(TEXT("FollowCamera"));
	FollowCamera->SetupAttachment(CameraBoom, USpringArmComponent::SocketName); // Attach the camera to the end of the boom and let the boom adjust to match the controller orientation
	FollowCamera->bUsePawnControlRotation = false; // Camera does not rotate relative to arm
	
	// Note: The skeletal mesh and anim blueprint references on the Mesh component (inherited from Character) 
	// are set in the derived blueprint asset named MyCharacter (to avoid direct content references in C++)
}

//////////////////////////////////////////////////////////////////////////
// Input

void ADTWProjectCharacter::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	// Set up gameplay key bindings
	check(PlayerInputComponent);
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ACharacter::Jump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ACharacter::StopJumping);
	PlayerInputComponent->BindAction("Record", IE_Pressed, this, &ADTWProjectCharacter::Record);

	PlayerInputComponent->BindAxis("MoveForward", this, &ADTWProjectCharacter::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &ADTWProjectCharacter::MoveRight);

	// We have 2 versions of the rotation bindings to handle different kinds of devices differently
	// "turn" handles devices that provide an absolute delta, such as a mouse.
	// "turnrate" is for devices that we choose to treat as a rate of change, such as an analog joystick
	PlayerInputComponent->BindAxis("Turn", this, &APawn::AddControllerYawInput);
	PlayerInputComponent->BindAxis("TurnRate", this, &ADTWProjectCharacter::TurnAtRate);
	PlayerInputComponent->BindAxis("LookUp", this, &APawn::AddControllerPitchInput);
	PlayerInputComponent->BindAxis("LookUpRate", this, &ADTWProjectCharacter::LookUpAtRate);

	// handle touch devices
	PlayerInputComponent->BindTouch(IE_Pressed, this, &ADTWProjectCharacter::TouchStarted);
	PlayerInputComponent->BindTouch(IE_Released, this, &ADTWProjectCharacter::TouchStopped);

	// VR headset functionality
	PlayerInputComponent->BindAction("ResetVR", IE_Pressed, this, &ADTWProjectCharacter::OnResetVR);
}

void ADTWProjectCharacter::OnResetVR()
{
	UHeadMountedDisplayFunctionLibrary::ResetOrientationAndPosition();
}

void ADTWProjectCharacter::TouchStarted(ETouchIndex::Type FingerIndex, FVector Location)
{
		Jump();
}

void ADTWProjectCharacter::TouchStopped(ETouchIndex::Type FingerIndex, FVector Location)
{
		StopJumping();
}

void ADTWProjectCharacter::TurnAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerYawInput(Rate * BaseTurnRate * GetWorld()->GetDeltaSeconds());
}

void ADTWProjectCharacter::LookUpAtRate(float Rate)
{
	// calculate delta for this frame from the rate information
	AddControllerPitchInput(Rate * BaseLookUpRate * GetWorld()->GetDeltaSeconds());
}

void ADTWProjectCharacter::MoveForward(float Value)
{
	if ((Controller != NULL) && (Value != 0.0f))
	{
		// find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// get forward vector
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::X);
		AddMovementInput(Direction, Value);
	}
}

void ADTWProjectCharacter::MoveRight(float Value)
{
	if ( (Controller != NULL) && (Value != 0.0f) )
	{
		// find out which way is right
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);
	
		// get right vector 
		const FVector Direction = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);
		// add movement in that direction
		AddMovementInput(Direction, Value);
	}
}

void ADTWProjectCharacter::Record(){
	if (recorded) {
		UE_LOG(LogTemp, Warning, TEXT("Stop recording"))
		recorded = false;
	}
	else {
		UE_LOG(LogTemp, Warning, TEXT("Start recording"))
		recorded = true;
	}

}

void ADTWProjectCharacter::FileIO() {
	FString AbsoluteFilePath = "C:/Users/ai-admin/Documents/Unreal Projects/DTWProject/movement.csv";			// Need to change filepath to desired location.
	
	const int64 FileSize = FPlatformFileManager::Get().GetPlatformFile().FileSize(*AbsoluteFilePath);
	UE_LOG(LogTemp, Warning, TEXT("Filesize: %d"), FileSize);

	if (FileSize <= 0) {			// If file is empty (0) or non-existent (-1), column names will be added/file with column names created.
		FFileHelper::SaveStringToFile(TEXT("loc_x,loc_y,loc_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,t,t_total,tick\r\n"), *AbsoluteFilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
	}

	float time_total = 0;			// Total time can be important if you need certain parts of a movement.
	for (int i = 0; i < xax.Num(); i++) {
		time_total += time[i];			
		FFileHelper::SaveStringToFile((FString::SanitizeFloat(xax[i]) + "," + FString::SanitizeFloat(yax[i]) + "," + FString::SanitizeFloat(zax[i]) + "," +
			FString::SanitizeFloat(xv[i+1]) + "," + FString::SanitizeFloat(yv[i+1]) + "," + FString::SanitizeFloat(zv[i+1]) + "," + FString::SanitizeFloat(xaccel[i]) + "," +
			FString::SanitizeFloat(yaccel[i]) + "," + FString::SanitizeFloat(zaccel[i]) + "," + FString::SanitizeFloat(time[i]) + "," + FString::SanitizeFloat(time_total) + "," + FString::FromInt(i+1) + "\r\n"),
			*AbsoluteFilePath, FFileHelper::EEncodingOptions::AutoDetect, &IFileManager::Get(), EFileWrite::FILEWRITE_Append);
	}
}

void ADTWProjectCharacter::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	Super::SetActorTickInterval(0.1);

	FVector ActorLocation = ADTWProjectCharacter::GetActorLocation();
	float loc_x = ActorLocation.GetComponentForAxis(EAxis::X);
	float loc_y = ActorLocation.GetComponentForAxis(EAxis::Y);
	float loc_z = ActorLocation.GetComponentForAxis(EAxis::Z);

	FVector ActorVelocity = ADTWProjectCharacter::GetVelocity();
	float vel_x = ActorVelocity.GetComponentForAxis(EAxis::X);
	float vel_y = ActorVelocity.GetComponentForAxis(EAxis::Y);
	float vel_z = ActorVelocity.GetComponentForAxis(EAxis::Z);

	//FTransform ActorTransform = ADTWProjectCharacter::GetActorTransform();
	//FTransform Yeet = ActorTransform.Inverse();
	//FString Test = Yeet.ToString();
	//UE_LOG(LogTemp, Warning, TEXT("%s"), *Test);
	
	if (recorded) {			

		if (xv.Num() == 0) {			// Adds a beginning velocity of 0 to all axis' to ensure the vel always begins at 0 (for plotting etc.).
			ADTWProjectCharacter::xv.Add(0.0);
			ADTWProjectCharacter::yv.Add(0.0);
			ADTWProjectCharacter::zv.Add(0.0);
		}

		ADTWProjectCharacter::xax.Add(loc_x);
		ADTWProjectCharacter::yax.Add(loc_y);
		ADTWProjectCharacter::zax.Add(loc_z);
		
		ADTWProjectCharacter::xv.Add(vel_x);
		ADTWProjectCharacter::yv.Add(vel_y);
		ADTWProjectCharacter::zv.Add(vel_z);
		
		ADTWProjectCharacter::time.Add(DeltaTime);
		 
		ADTWProjectCharacter::xaccel.Add(((xv[xv.Num() - 1] - xv[xv.Num() - 2]) / DeltaTime));
		ADTWProjectCharacter::yaccel.Add(((yv[yv.Num() - 1] - yv[yv.Num() - 2]) / DeltaTime));
		ADTWProjectCharacter::zaccel.Add(((zv[zv.Num() - 1] - zv[zv.Num() - 2]) / DeltaTime));
	}

	if (!recorded && time.Num() != 0) {			
		UE_LOG(LogTemp, Warning, TEXT("Recorded Movement"))
		FileIO();
		xax.Empty();
		yax.Empty();
		zax.Empty();
		xv.Empty();
		yv.Empty();
		zv.Empty();
		xaccel.Empty();
		yaccel.Empty();
		zaccel.Empty();
		time.Empty();
		recorded = false;
	}
}