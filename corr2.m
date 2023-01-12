function R = corr2(X,Y)
fenmu1 = X-mean(mean(X));
fenmu1=fenmu1.^2;
fenmu1 = sum(sum(fenmu1));

fenmu2 = Y-mean(mean(Y));
fenmu2=fenmu2.^2;
fenmu2 = sum(sum(fenmu2));
fenzi = (X-mean(mean(X))).*(Y-mean(mean(Y)));
fenzi = sum(sum(fenzi));
R = fenzi/sqrt(fenmu1*fenmu2);
end