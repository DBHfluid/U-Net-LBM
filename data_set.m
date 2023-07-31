clear all;
close all;
for a =1:2010

NX=64; 
NY=32;

OMEGA=1.0; 
rho0=1.0; 
deltaUX=10^-6;

W=[4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36];

cx=[0,0,1,1, 1, 0,-1,-1,-1];
cy=[0,1,1,0,-1,-1,-1, 0, 1];

SOLID=rand(NX*NY,1)>.9;
SOLID(1:NX,1)=1;
SOLID(NX*NY-NX:NX*NY,1)=1;

N=bsxfun(@times,rho0*ones(NX*NY,9),W);

for t_=1:8000
    for i=2:9
        N(:,i)=reshape(circshift(reshape(N(:,i),NX,NY),[cx(i),cy(i)]),NX*NY,1);
    end
    N_SOLID=N(SOLID,[1 6 7 8 9 2 3 4 5]); % Bounce Back and No Collision
    rho = sum(N,2);
    ux  = sum(bsxfun(@times,N,cx),2)./rho;
    ux=ux+deltaUX;
    uy  = sum(bsxfun(@times,N,cy),2)./rho;
    workMatrix=ux*cx+uy*cy;
    workMatrix=(3+4.5*workMatrix).*workMatrix;
    workMatrix=bsxfun(@minus,workMatrix,1.5*(ux.^2+uy.^2));
    workMatrix=bsxfun(@times,1+workMatrix,W);
    workMatrix=bsxfun(@times,workMatrix,rho);
    N=N+(workMatrix-N)*OMEGA;
    N(SOLID,:)=N_SOLID;
end
ux(SOLID)=0; 
uy(SOLID)=0;
ux=reshape(ux,NX,NY)';
uy=reshape(uy,NX,NY)';
figure(1);
clf;
hold on;
colormap(gray(2));
image(2-reshape(SOLID,NX,NY)');
quiver(1:NX,1:NY,ux,uy,1.5,'b');
axis([0.5 NX+0.5 0.5 NY+0.5]);
axis image;
title(['Velocity field after ',num2str(t_),' time steps']);

u=sqrt(ux.*ux + uy.*uy);
d=0;
B = u';

B = B(:)';
B=B(B~=d);
ma=max(B);
mi=min(B);
I2=mat2gray(flipud(u),[mi ma]);



%I=mat2gray(flipud(u));
I1=~(flipud((reshape(SOLID,NX,NY))'));



name1=['C:\Users\DESHBANDHU\Desktop\New folder (8)\folder' num2str(a),'\images'];%new folder name
mkdir(name1)%create new folder
imwrite(I1,fullfile(name1,strcat('folder',num2str(a),'.png')))

name2=['C:\Users\DESHBANDHU\Desktop\New folder (8)\folder' num2str(a),'\maskss'];%new folder name
mkdir(name2)%create new folder
imwrite(I2,fullfile(name2,strcat('folder',num2str(a),'.png')))

%name=['C:\Users\DESHBANDHU\Desktop\New folder (7)\folder' ];%new folder name
%mkdir(name)%create new folder
%imwrite(iii,fullfile(name,strcat('folder1','.png')))
%imwrite(II,fullfile(name,strcat('folder2','.png')))

end

clear all;
close all;

